#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <seal/seal.h>
#include "HomomorphicLogisticRegression.h"


using namespace std;
using namespace seal;

// ========== UTILITIES ==========





vector<vector<double>> load_csv(const string &filename)
{
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
    while (getline(file, line))
    {
        vector<double> row;
        stringstream ss(line);
        string value;
        while (getline(ss, value, ','))
            row.push_back(stod(value));
        data.push_back(row);
    }
    return data;
}

vector<double> load_csv_1d(const string &filename)
{
    vector<double> result;
    ifstream file(filename);
    string line;
    while (getline(file, line))
        result.push_back(stod(line));
    return result;
}

// ========== MAIN ==========

int main()
{
    cout << "=== Encrypted Binary Logistic Regression using CKKS ===\n";

    auto W = load_csv("../../data/result_binary_logreg/weight_vector.csv");
    auto b = load_csv_1d("../../data/result_binary_logreg/bias_scalar.csv")[0];
    auto X_test = load_csv("../../data/result_binary_logreg/X_test.csv");
    auto y_test = load_csv_1d("../../data/result_binary_logreg/y_test.csv");

    size_t N = X_test.size();
    size_t dim = W[0].size();
    vector<double> w = W[0];

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 60}));
    auto context = make_shared<SEALContext>(parms);

    KeyGenerator keygen(*context);
    PublicKey pk;
    keygen.create_public_key(pk);
    RelinKeys rlk;
    keygen.create_relin_keys(rlk);
    GaloisKeys gk;
    keygen.create_galois_keys(gk);

    Encryptor encryptor(*context, pk);
    Evaluator evaluator(*context);
    Decryptor decryptor(*context, keygen.secret_key());
    CKKSEncoder encoder(*context);

    double scale = pow(2.0, 40);
    size_t correct = 0;

    for (size_t i = 0; i < N; i++)
    {
        vector<double> x_slots(encoder.slot_count(), 0.0);
        for (size_t j = 0; j < dim; j++) x_slots[j] = X_test[i][j];

        Plaintext pt_x;
        encoder.encode(x_slots, scale, pt_x);
        Ciphertext ct_x;
        encryptor.encrypt(pt_x, ct_x);

        Plaintext pt_w;
        encoder.encode(w, scale, pt_w);
        Ciphertext ct_prod;
        evaluator.multiply_plain(ct_x, pt_w, ct_prod);
        evaluator.relinearize_inplace(ct_prod, rlk);
        evaluator.rescale_to_next_inplace(ct_prod);
        ct_prod.scale() = scale;

        for (int step = 1; step < (int)dim; step <<= 1)
        {
            Ciphertext tmp;
            evaluator.rotate_vector(ct_prod, step, gk, tmp);
            evaluator.add_inplace(ct_prod, tmp);
        }

        Plaintext pt_b;
        encoder.encode(b, ct_prod.scale(), pt_b);
        evaluator.mod_switch_to_inplace(pt_b, ct_prod.parms_id());
        evaluator.add_plain_inplace(ct_prod, pt_b);

        Ciphertext ct_sigmoid = sigmoid_quadratic(
            ct_prod, 0.5, 0.25, -0.015,
            evaluator, encoder, rlk, context, scale
        );

        double prob = decrypt_first_slot(ct_sigmoid, decryptor, encoder);
        int pred = prob >= 0.5 ? 1 : 0;
        int true_label = static_cast<int>(y_test[i]);

        if (pred == true_label) correct++;

        cout << "Sample " << i << ": prob=" << prob
             << ", pred=" << pred << ", label=" << true_label << endl;
    }

    cout << "\nEncrypted accuracy = " << static_cast<double>(correct) / N << endl;
    return 0;
}