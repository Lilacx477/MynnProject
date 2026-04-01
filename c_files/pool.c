#if defined(_WIN32) || defined(_WIN64)
#define API __declspec(dllexport)
#else
#define API
#endif

#ifdef __cplusplus
extern "C" {
#endif

API void maxpool(
    const float* input, //入力テンソル(2)のポインタ
    float* output, //確保した出力テンソル(4)のポインタ
    float* index, //確保したインデックステンソル(4)のポインタ
    int N,
    int C, int H, int W,
    int H_out, int W_out,
    int H_win, int W_win
) {
    for (int n=0; n < N; n++) {
        for (int ho=0; ho < H_out; ho++) {

            for (int wo=0; wo < W_out; wo++) {

                for (int c=0; c < C; c++) {
                    int row = n * H_out * W_out + ho * W_out + wo;
                    int idx0 = row * (C * H_win * W_win) + c * (H_win * W_win);
                    float max = input[idx0];
                    int ih0 = ho * H_win;
                    int iw0 = wo * W_win;
                    int id = ((n * C + c) * H + ih0) * W + iw0;

                    for (int hw=0; hw < H_win; hw++) {

                        for (int ww=0; ww < W_win; ww++) {
                            
                            int idx_in = row * (C * H_win * W_win)
                                + c * (H_win * W_win)
                                + hw * W_win
                                + ww;
                            int ih = ho * H_win + hw;
                            int iw = wo * W_win + ww;
                            int orig_idx = ((n*C +c) * H + ih) * W + iw;
                            if (max < input[idx_in]) {
                                max = input[idx_in];
                                id = orig_idx;
                            }
                        }
                    }
                    output[W_out*(H_out*(C*n + c) + ho) + wo] = max;
                    index[W_out*(H_out*(C*n + c) + ho) + wo] = id;
                }
            }
        }
    }
}


API void avepool(
    const float* input, //入力テンソル(2)のポインタ
    float* output, //確保した出力テンソル(3)のポインタ
    int N,
    int C, int H_out, int W_out,
    int H_win, int W_win
) {
    for (int n=0; n < N; n++) {
        for (int ho=0; ho < H_out; ho++) {

            for (int wo=0; wo < W_out; wo++) {

                for (int c=0; c < C; c++) {
                    int row = n * H_out * W_out + ho * W_out + wo;
                    float sum = 0.0;

                    for (int hw=0; hw < H_win; hw++) {

                        for (int ww=0; ww < W_win; ww++) {
                        int idx_in = row * (C * H_win * W_win)
                            + c * (H_win * W_win)
                            + hw * W_win
                            + ww;
                        sum += input[idx_in];
                    }
                }
                output[W_out*(H_out*(C*n + c) + ho) + wo] = sum / (H_win*W_win);
                }
            }
        }
    }
}

API void grad_maxpool(
    const float* input, //入力テンソル(3)のポインタ
    float* output, //確保した出力テンソル(3)のポインタ
    const float* index, //入力インデックステンソル(3)のポインタ
    int N,
    int C, int H, int W,
    int H_out, int W_out
) {
    for (int n=0; n < N; n++) {
        for (int c=0; c < C; c++)
        for (int ho=0; ho < H_out; ho++) {
            for (int wo=0; wo < W_out; wo++) {
                int idx = W_out*(H_out*(C*n + c) + ho) + wo;
                int idx_out = (int)index[idx];
                output[idx_out] = input[idx];
            }
        }
    }
}


API void tangent_maxpool(
    const float* input, //入力テンソル(3)のポインタ
    float* output, //確保した出力テンソル(3)のポインタ
    const float* index, //入力インデックステンソル(3)のポインタ
    int N,
    int C, int H, int W,
    int H_out, int W_out
) {
    for (int n=0; n < N; n++) {
        for (int c=0; c < C; c++) {
            for (int ho=0; ho < H_out; ho++) {
                for (int wo=0; wo < W_out; wo++) {
                int idx = W_out*(H_out*(C*n + c) + ho) + wo;
                int idx_out = (int)index[idx];
                output[idx] = input[idx_out];
                }
            }
        }
    }
}

API void grad_avepool(
    const float* input, //入力テンソル(3)のポインタ
    float* output, //確保した出力テンソル(3)のポインタ
    int N,
    int C, int H, int W,
    int H_out, int W_out,
    int H_win, int W_win ) {
    for (int n=0; n < N; n++) {
        for (int c=0; c < C; c++) {
            for (int h=0; h < H_win*H_out; h++) {
                for (int w=0; w < W_win*W_out; w++) {
                    int ho = h / H_win;
                    int wo = w / W_win;
                    int idx_in = W_out*(H_out*(C*n + c) + ho) + wo;
                    int idx_out = W*(H*(C*n + c) + h) + w;
                    output[idx_out] = input[idx_in] / (float)(H_win*W_win);
                    }
                }
            }
        }
    }
