#ifndef HOMOMORPHIC_LOGISTIC_REGRESSION_H
#define HOMOMORPHIC_LOGISTIC_REGRESSION_H

#include <vector>
#include <string>
#include <seal/seal.h>
#include <memory>

using namespace std;
using namespace seal;


map<string, string> load_env(const string& filename);
// CSV Utils
vector<vector<double>> read_csv_matrix(const string &filename, size_t cols = 0);
vector<double> read_csv_single_col(const string &filename);

// SEAL Utils
void print_parameters(const SEALContext &context);
double decrypt_first_slot(const Ciphertext &ct, Decryptor &decryptor, CKKSEncoder &encoder);
void align_levels_and_scales(Ciphertext &ctA, Ciphertext &ctB, Evaluator &evaluator, shared_ptr<SEALContext> context);
void align_plain_to_cipher(Plaintext &pt, Ciphertext &ct, Evaluator &evaluator);
void rotate_and_sum(Ciphertext &ct_in, size_t input_dim, Evaluator &evaluator, const GaloisKeys &gal_keys);

#endif // HOMOMORPHIC_LOGISTIC_REGRESSION_H