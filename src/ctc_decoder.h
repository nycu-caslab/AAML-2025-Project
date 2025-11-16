/*
 * A simple, header-only CTC Greedy Decoder.
 * This is not a full "beam search" (which is complex),
 * but it's small, fast, and will give you the correct text.
 */
 #ifndef CTC_DECODER_H_
 #define CTC_DECODER_H_
 
 #include <string.h>
 
 // This defines the alphabet from the model's 'get_label_dict.py' script.
 // "abcdefghijklmnopqrstuvwxyz' @" (29 characters total)
 // The 28th index is the '@' symbol, which is the "blank" token.
 const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz' @";
 const int kBlankToken = 28;
 const int kNumClasses = 29;
 
 /**
  * @brief Decodes the raw output tensor from Wav2letter using a greedy algorithm.
  *
  * @param output_tensor Pointer to the start of the (1, 1, 148, 29) output tensor.
  * @param result_buffer A char buffer to write the final string into.
  * @param buffer_len The max size of the result_buffer.
  */
 inline void ctc_greedy_decoder(int8_t* output_tensor, char* result_buffer,
                                size_t buffer_len) {
   // Output shape from README is (1, 1, 148, 29)
   const int num_timesteps = 148;
 
   int prev_token = -1;
   size_t buffer_pos = 0;
 
   // Iterate over each of the 148 timesteps
   for (int t = 0; t < num_timesteps; ++t) {
     // Find the class (0-28) with the highest score at this timestep
     int8_t* timestep_scores = output_tensor + (t * kNumClasses);
     int8_t max_score = -128;
     int max_index = -1;
 
     for (int c = 0; c < kNumClasses; ++c) {
       if (timestep_scores[c] > max_score) {
         max_score = timestep_scores[c];
         max_index = c;
       }
     }
 
     // Now, apply CTC greedy logic:
     // 1. If it's not blank and not the same as the previous token,
     //    we add it to our string.
     if (max_index != kBlankToken && max_index != prev_token) {
       if (buffer_pos < buffer_len - 1) {  // -1 for null terminator
         result_buffer[buffer_pos++] = kAlphabet[max_index];
       }
     }
     // 2. If it's the "blank" token, we reset our 'prev_token'
     if (max_index == kBlankToken) {
       prev_token = -1;
     } else {
       prev_token = max_index;
     }
   }
 
   // Add the null terminator to make it a valid C-string
   result_buffer[buffer_pos] = '\0';
 }
 
 #endif  // CTC_DECODER_H_