#if defined(_WIN32) || defined(_WIN64)
#define API __declspec(dllexport)
#else
#define API
#endif

API void im2col(
    const float* input, //入力テンソル(4)のポインタ
    float* output, //確保した出力テンソル(2)のポインタ
    int N, //バッチサイズ
    int H, int W,
    int H_out, int W_out,
    int C, int Kh, int Kw,
    int Sh, int Sw
) {
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {

            int h_base = Sh*h_out;

            for (int w_out = 0; w_out < W_out; w_out++) {
            
                int w_base = Sw*w_out;

                for (int c = 0; c < C; c++){

                    for (int kh = 0; kh < Kh; kh++) {

                        for (int kw = 0; kw < Kw; kw++) {

                            int ih = h_base + kh;

                            int iw = w_base + kw;

                            int idx_in =  ((n*C +c)*H + ih)*W + iw;

                            int col = (c*Kh + kh)*Kw + kw;

                            int row = n*H_out*W_out + h_out*W_out + w_out;

                            int idx_out = row*(C*Kh*Kw) + col;

                            output[idx_out] = input[idx_in];

                        }
                    }
                }
            }
        }
    }
}

API void col2im(
    const float* input, //入力テンソル(2)のポインタ
    float* output, //確保した出力テンソル(4)のポインタ
    int N, //バッチサイズ
    int H, int W,
    int H_out, int W_out,
    int C, int Kh, int Kw,
    int Sh, int Sw
) {
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {

            int h_base = Sh*h_out;

            for (int w_out = 0; w_out < W_out; w_out++) {
            
                int w_base = Sw*w_out;

                for (int c = 0; c < C; c++){

                    for (int kh = 0; kh < Kh; kh++) {

                        for (int kw = 0; kw < Kw; kw++) {

                            int ih = h_base + kh;

                            int iw = w_base + kw;

                            int idx_out =  ((n*C + c)*H + ih)*W + iw;

                            int col = (c*Kh + kh)*Kw + kw;

                            int row = n*H_out*W_out + h_out*W_out + w_out;

                            int idx_in = row*(C*Kh*Kw) + col;

                            output[idx_out] += input[idx_in];

                        }
                    }
                }
            }
        }
    }
}

API void reshape_4to2(
    const float* input, //入力テンソル(4)のポインタ
    float* output, //確保した出力テンソル(2)のポインタ
    int K, int C, int Kh, int Kw
) {
    for (int k = 0; k < K; k++) {
        
        for (int c = 0; c < C; c++) {

            for (int kh = 0; kh < Kh; kh++) {

                for (int kw = 0; kw < Kw; kw++) {
                    
                    int idx = ((C*k + c)*Kh + kh)*Kw + kw;

                    output[idx] = input[idx];

                }
            }
        }
    }
}

API void reshape_3to2(
    const float* input, //入力テンソル(3)のポインタ
    float* output, //確保した出力テンソル(2)のポインタ
    int K, int H_out, int W_out
) {
    for (int k = 0; k < K; k++) {
        
        for (int h = 0; h < H_out; h++) {

            for (int w = 0; w < W_out; w++) {
                    
                int idx = W_out*(H_out*k + h) + w;

                output[idx] = input[idx];

            }
        }
    }
}

API void reshape_2to3(
    const float* input, //入力テンソル(2)のポインタ
    float* output, //確保した出力テンソル(3)のポインタ
    int K, int H_out, int W_out
) {
    for (int k = 0; k < K; k++) {

        for (int h = 0; h < H_out; h++) {

            for (int w = 0; w < W_out; w++) {

                int idx = (H_out*k + h)*W_out + w;

                output[idx] = input[idx];

            }
        }
    }
}

API void reshape_2to4(
    const float* input, //入力テンソル(2)のポインタ
    float* output, //確保した出力テンソル(4)のポインタ
    int K, int C, int Kh, int Kw
) {
    for (int k = 0; k < K; k++) {

        for (int c = 0; c < C; c++) {

            for (int kh = 0; kh < Kh; kh++) {

                for (int kw = 0; kw < Kw; kw++) {

                    int idx = Kw*(Kh*(C*k+ c) + kh) + kw;

                    output[idx] = input[idx];

                }
            }
        }
    }
}
