/*
 * Copyright 2021 The CFU-Playground Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include "proj_menu.h"

 #include <stdio.h>
 
 #include "cfu.h"
 #include "menu.h"
 #include "wav2letter/wav2letter.h" // Your existing include for 'w'
 
 // <<< MODIFIED: Add MLPerf includes >>>
 #include "third_party/mlperf_tiny/api/internally_implemented.h"
 #include "third_party/mlperf_tiny/api/submitter_implemented.h"
 
 namespace {
 
 // <<< MODIFIED: Add this function >>>
 // This function enters the MLPerf benchmark mode and never returns.
 void do_enter_mlperf_tiny(void){
   // Initialize the benchmark harness
   ee_benchmark_initialize(); 
   th_getchar();
   // This is the forever-loop that processes serial commands
   while (1) {
     int c;
     c = th_getchar(); // Get char from serial
     #ifndef MLPERF_TINY_NO_ECHO
       putchar(c); // Echo back if not disabled
     #endif
     ee_serial_callback(c); // Feed the char to the MLPerf parser
   }
 }
 
 struct Menu MENU = {
     "Project Menu",
     "project",
     {
         // Your existing menu item for human-testing
         MENU_ITEM('w', "Wav2letter Tests", wav2letter_pruned_menu),
 
         // <<< MODIFIED: Add this new menu item >>>
         // 'b' for "benchmark"
         MENU_ITEM('b', "Enter MLPerf Tiny Benchmark Interface", do_enter_mlperf_tiny),
         
         MENU_END,
     },
 };
 
 };  // anonymous namespace
 
 extern "C" void do_proj_menu() { menu_run(&MENU); }