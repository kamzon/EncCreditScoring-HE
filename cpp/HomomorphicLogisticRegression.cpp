#include "HomomorphicLogisticRegression.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace std;
using namespace seal;

#include <map>

// Utility function to load .env file
map<string, string> load_env(const string& filename)
{
    map<string, string> env_map;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error: Could not open .env file: " << filename << endl;
        return env_map;
    }
    
    string line;
    while (getline(file, line))
    {
        if (line.empty() || line[0] == '#') continue; // Skip empty lines and comments
        auto delimiter_pos = line.find('=');
        if (delimiter_pos == string::npos) continue; // Skip invalid lines
        string key = line.substr(0, delimiter_pos);
        string value = line.substr(delimiter_pos + 1);
        env_map[key] = value;
    }
    file.close();
    return env_map;
}

// CSV Utils
vector<vector<double>> read_csv_matrix(const string &filename, size_t cols)
{
    vector<vector<double>> data;
    ifstream ifs(filename);
    if (!ifs.is_open())
    {
        cerr << "Error: cannot open file " << filename << endl;
        return data;
    }
    string line;
    while (getline(ifs, line))
    {
        if (line.empty()) continue;
        vector<double> row;
        stringstream ss(line);
        string val;
        while (getline(ss, val, ','))
        {
            row.push_back(stod(val));
        }
        if (cols == 0 || row.size() == cols)
            data.push_back(row);
    }
    ifs.close();
    return data;
}

vector<double> read_csv_single_col(const string &filename)
{
    vector<double> data;
    ifstream ifs(filename);
    if (!ifs.is_open())
    {
        cerr << "Error: cannot open file " << filename << endl;
        return data;
    }
    string line;
    while (getline(ifs, line))
    {
        if (line.empty()) continue;
        data.push_back(stod(line));
    }
    ifs.close();
    return data;
}

// SEAL Utils
void print_parameters(const SEALContext &context)
{
    auto &context_data = *context.key_context_data();
    auto &parms = context_data.parms();
    cout << "Encryption parameters:" << endl;
    cout << "  poly_modulus_degree: " << parms.poly_modulus_degree() << endl;
    cout << "  coeff_modulus size: "
         << context_data.total_coeff_modulus_bit_count() << " bits" << endl;
}

double decrypt_first_slot(const Ciphertext &ct, Decryptor &decryptor, CKKSEncoder &encoder)
{
    Plaintext pt;
    decryptor.decrypt(ct, pt);
    vector<double> decoded;
    encoder.decode(pt, decoded);
    if (!decoded.empty()) {
        return decoded[0];
    }
    return 0.0;
}

void align_levels_and_scales(
    Ciphertext &ctA,
    Ciphertext &ctB,
    Evaluator &evaluator,
    shared_ptr<SEALContext> context
)
{
    while (ctA.parms_id() != ctB.parms_id())
    {
        auto idxA = context->get_context_data(ctA.parms_id())->chain_index();
        auto idxB = context->get_context_data(ctB.parms_id())->chain_index();
        if (idxA > idxB)
            evaluator.mod_switch_to_next_inplace(ctA);
        else
            evaluator.mod_switch_to_next_inplace(ctB);
    }
    double diff = fabs(log2(ctA.scale()) - log2(ctB.scale()));
    if (diff > 1.0)
    {
        if (ctA.scale() > ctB.scale())
            ctA.scale() = ctB.scale();
        else
            ctB.scale() = ctA.scale();
    }
}

void align_plain_to_cipher(Plaintext &pt, Ciphertext &ct, Evaluator &evaluator, shared_ptr<SEALContext> context) {
    evaluator.mod_switch_to_inplace(pt, ct.parms_id());
    double diff = fabs(log2(ct.scale()) - log2(pt.scale()));
    if (diff > 1.0) {
        if (ct.scale() > pt.scale()) ct.scale() = pt.scale();
        else pt.scale() = ct.scale();
    }
}


void rotate_and_sum(
    Ciphertext &ct_in,
    size_t input_dim,
    Evaluator &evaluator,
    const GaloisKeys &gal_keys
)
{
    size_t step = 1;
    while (step < input_dim)
    {
        Ciphertext tmp;
        evaluator.rotate_vector(ct_in, step, gal_keys, tmp);
        evaluator.add_inplace(ct_in, tmp);
        step <<= 1;
    }
}

Ciphertext sigmoid_quadratic(
    Ciphertext ct_z,
    double a0, double a1, double a2,
    Evaluator &evaluator,
    CKKSEncoder &encoder,
    RelinKeys &relin_keys,
    shared_ptr<SEALContext> context,
    double scale)
{
    Ciphertext z2;
    evaluator.square(ct_z, z2);
    evaluator.relinearize_inplace(z2, relin_keys);
    evaluator.rescale_to_next_inplace(z2);
    z2.scale() = scale;

    Plaintext pt_a1, pt_a2, pt_a0;
    encoder.encode(a1, scale, pt_a1);
    encoder.encode(a2, scale, pt_a2);
    encoder.encode(a0, scale, pt_a0);

    evaluator.mod_switch_to_inplace(pt_a1, ct_z.parms_id());
    Ciphertext a1z;
    evaluator.multiply_plain(ct_z, pt_a1, a1z);
    evaluator.relinearize_inplace(a1z, relin_keys);
    evaluator.rescale_to_next_inplace(a1z);
    a1z.scale() = scale;

    evaluator.mod_switch_to_inplace(pt_a2, z2.parms_id());
    Ciphertext a2z2;
    evaluator.multiply_plain(z2, pt_a2, a2z2);
    evaluator.relinearize_inplace(a2z2, relin_keys);
    evaluator.rescale_to_next_inplace(a2z2);
    a2z2.scale() = scale;

    while (a1z.parms_id() != a2z2.parms_id())
    {
        if (context->get_context_data(a1z.parms_id())->chain_index() >
            context->get_context_data(a2z2.parms_id())->chain_index())
            evaluator.mod_switch_to_next_inplace(a1z);
        else
            evaluator.mod_switch_to_next_inplace(a2z2);
    }
    evaluator.add_inplace(a1z, a2z2);

    align_plain_to_cipher(pt_a0, a1z, evaluator, context);
    evaluator.add_plain_inplace(a1z, pt_a0);
    return a1z;
}
