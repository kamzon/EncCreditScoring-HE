#include "HomomorphicLogisticRegression.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace seal;

int main()
{
    auto t_start_all = chrono::high_resolution_clock::now();

    // ========== 1) Load CSV Data ==========
    cout << "=== 1) Loading CSV files for weights, biases, X_test, y_test ===" << endl;
    vector<vector<double>> W = read_csv_matrix("/Users/noureddinekamzon/Desktop/Uni passau/master theisi/EncCreditScoring-HE/data/weight_matrix.csv");
    vector<vector<double>> b_mat = read_csv_matrix("/Users/noureddinekamzon/Desktop/Uni passau/master theisi/EncCreditScoring-HE/data/bias_vector.csv");
    vector<vector<double>> X_test = read_csv_matrix("/Users/noureddinekamzon/Desktop/Uni passau/master theisi/EncCreditScoring-HE/data/X_test.csv");
    vector<double> y_test = read_csv_single_col("/Users/noureddinekamzon/Desktop/Uni passau/master theisi/EncCreditScoring-HE/data/y_test.csv");

    if (b_mat.size() != 1)
    {
        cerr << "Error: bias_vector.csv should have exactly 1 row => (1 x num_classes)." << endl;
        return -1;
    }
    vector<double> b = b_mat[0];
    size_t num_classes = W.size();
    if (num_classes == 0)
    {
        cerr << "Error: weight_matrix.csv is empty." << endl;
        return -1;
    }
    size_t input_dim = W[0].size();
    for (size_t i = 1; i < num_classes; i++)
    {
        if (W[i].size() != input_dim)
        {
            cerr << "Error: weight_matrix row " << i << " dimension mismatch." << endl;
            return -1;
        }
    }
    size_t N = X_test.size();
    if (N == 0 || y_test.size() != N)
    {
        cerr << "Error: mismatch in X_test rows vs. y_test or files empty." << endl;
        return -1;
    }

    cout << "  Loaded weight_matrix: " << num_classes << " rows, each with "
         << input_dim << " columns.\n";
    cout << "  Loaded bias_vector of length " << b.size() << ".\n";
    cout << "  Loaded X_test: " << N << " rows, " << input_dim << " columns.\n";
    cout << "  Loaded y_test: " << y_test.size() << " labels.\n\n";

    // ========== 2) Setup CKKS Context, Keys ==========
    cout << "=== 2) Setting up SEAL (CKKS) context ===" << endl;
    auto t_seal_setup_0 = chrono::high_resolution_clock::now();

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 40, 40, 60}));
    auto context = make_shared<SEALContext>(parms);

    print_parameters(*context);
    cout << endl;

    KeyGenerator keygen(*context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    Encryptor encryptor(*context, public_key);
    Evaluator evaluator(*context);
    Decryptor decryptor(*context, secret_key);
    CKKSEncoder encoder(*context);

    double init_scale = pow(2.0, 40);

    auto t_seal_setup_1 = chrono::high_resolution_clock::now();
    auto seal_setup_time = chrono::duration_cast<chrono::milliseconds>(t_seal_setup_1 - t_seal_setup_0).count();
    cout << "SEAL setup took: " << seal_setup_time << " ms\n" << endl;

    // ========== 3) Homomorphic Inference + Plaintext Comparison ==========
    cout << "=== 3) Homomorphic inference on test set ===" << endl;
    cout << "We'll compute z_i = w_i.x + b_i for each class i, comparing plaintext vs. encrypted.\n"
         << "Then we pick argmax(encrypted logits) as the predicted class.\n\n";

    auto t_inference_start = chrono::high_resolution_clock::now();
    size_t correct_count = 0;
    size_t limit_samples = N;
    cout << "Processing " << limit_samples << " samples...\n";

    for (size_t sample_idx = 0; sample_idx < limit_samples; sample_idx++)
    {
        cout << "\n--- Sample #" << sample_idx << " ---\n";

        const auto &xrow = X_test[sample_idx];
        vector<double> logits_plain(num_classes, 0.0);
        for (size_t i = 0; i < num_classes; i++)
        {
            double dot = 0.0;
            for (size_t j = 0; j < input_dim; j++)
            {
                dot += W[i][j] * xrow[j];
            }
            logits_plain[i] = dot + b[i];
        }

        auto t_sample_start = chrono::high_resolution_clock::now();
        vector<double> xslots(encoder.slot_count(), 0.0);
        for (size_t j = 0; j < input_dim; j++)
        {
            xslots[j] = xrow[j];
        }

        Plaintext pt_x;
        encoder.encode(xslots, init_scale, pt_x);

        Ciphertext ct_x;
        encryptor.encrypt(pt_x, ct_x);

        vector<double> logits_enc(num_classes, 0.0);

        for (size_t i = 0; i < num_classes; i++)
        {
            Plaintext pt_w;
            encoder.encode(W[i], init_scale, pt_w);

            Ciphertext ct_prod;
            evaluator.multiply_plain(ct_x, pt_w, ct_prod);
            evaluator.relinearize_inplace(ct_prod, relin_keys);
            evaluator.rescale_to_next_inplace(ct_prod);
            ct_prod.scale() = init_scale;

            rotate_and_sum(ct_prod, input_dim, evaluator, galois_keys);

            Plaintext pt_b;
            encoder.encode(b[i], init_scale, pt_b);
            align_plain_to_cipher(pt_b, ct_prod, evaluator);
            evaluator.add_plain_inplace(ct_prod, pt_b);

            double val_enc = decrypt_first_slot(ct_prod, decryptor, encoder);
            logits_enc[i] = val_enc;
        }

        cout << " Class | Plaintext_z    | Encrypted_z     | Diff (enc-plain)\n";
        cout << "-------+----------------+-----------------+------------------\n";
        for (size_t i = 0; i < num_classes; i++)
        {
            double diff = logits_enc[i] - logits_plain[i];
            cout << "   " << i
                 << "   | " << logits_plain[i]
                 << "   | " << logits_enc[i]
                 << "   | " << diff << "\n";
        }

        int predicted_class = 0;
        double max_val = logits_enc[0];
        for (int i = 1; i < (int)num_classes; i++)
        {
            if (logits_enc[i] > max_val)
            {
                max_val = logits_enc[i];
                predicted_class = i;
            }
        }

        int true_label = static_cast<int>(y_test[sample_idx]);
        bool correct = (predicted_class == true_label);
        if (correct) correct_count++;

        auto t_sample_end = chrono::high_resolution_clock::now();
        auto sample_time = chrono::duration_cast<chrono::milliseconds>(t_sample_end - t_sample_start).count();

        cout << " => Predicted class = " << predicted_class << ", True label = " << true_label
             << (correct ? " (correct)" : " (wrong)") << "\n";
        cout << " Sample #" << sample_idx << " total time = " << sample_time << " ms\n";
    }

    double accuracy = double(correct_count) / double(limit_samples);
    auto t_inference_end = chrono::high_resolution_clock::now();
    auto inference_time = chrono::duration_cast<chrono::milliseconds>(t_inference_end - t_inference_start).count();

    cout << "\n=== Final Results ===\n";
    cout << "Processed " << limit_samples << " samples.\n";
    cout << "Encrypted multi-class classification accuracy = " << accuracy << "\n";
    cout << "Total inference time = " << inference_time << " ms\n";

    auto t_end_all = chrono::high_resolution_clock::now();
    auto total_time = chrono::duration_cast<chrono::milliseconds>(t_end_all - t_start_all).count();
    cout << "Overall program runtime = " << total_time << " ms\n";

    return 0;
}