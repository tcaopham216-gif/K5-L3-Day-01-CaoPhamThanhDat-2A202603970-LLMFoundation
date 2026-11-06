# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603970
- **Họ và tên:** Cao Phạm Thành Đạt
- **SEED:** 5165

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

| Cách | F1 test (toàn bộ) | F1 SUB 40 | F1 50 case khó | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.855 | 0.919 | 0.343 | ~0.3 | 0 |
| 2. BoW + MLP | 0.918 | 0.974 | 0.324 | ~0.6 | 0 |
| 3a. Encoder + kNN | 0.946 | 1.000 | 0.737 | ~164 | 0 |
| 3b. Encoder + LogReg | 0.954 | 1.000 | 0.596 | ~157 | 0 |
| 3c. Encoder + MLP head | 0.959 | 1.000 | 0.653 | ~39 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0.900 | 0.983 | ~1,500 | 0.006 |
| 4c. LLM nhỏ few-shot | — | — | — | ~1,500 | 0.017 |
| 4b. LLM lớn zero-shot | — | — | — | ~1,500 | 0.062 |

Ghi chú: p95 của 3a/3b/3c đo khi encode từng tin, nên có thể bị nhiễu vì máy đang bận. Các cột F1 SUB và 50 case khó lấy từ bảng `RESULTS` và bảng stress test trong notebook.

## 3. Dự đoán và thực tế

1. **Thí nghiệm 3.B (embedding tìm theo nghĩa / LogReg):** em đoán Encoder + LogReg đúng nhiều hơn BoW. Thực tế hoà 5/6. Cả hai đều sai tin "Last chance! Your account rewards expire tonight…", vì tin này không có từ khoá và giống thông báo tài khoản thật. Em đánh giá quá cao lợi thế của embedding trên một mẫu 6 tin.

2. **Bài toán 50 case khó (`predict("hard_eval", …)`):** em đoán BoW + LogReg đạt accuracy khoảng 0.70 và LLM nhỏ khoảng 0.80. Thực tế BoW + LogReg chỉ đạt 0.54 và LLM nhỏ đạt 0.98. Em đánh giá quá thấp LLM và quá cao BoW. Phần sai lớn nhất là BoW: các tin né từ khoá và giọng tự nhiên gần như không nhận ra.

3. **Đường Pareto (`predict("tradeoff", …)`):** em đoán Pareto gồm 3b và 3a. Thực tế Pareto gồm BoW + LogReg, BoW + MLP và Encoder + MLP head. Em nhầm vì chỉ nhìn F1 mà không để ý đến p95: kNN và LogReg trên embedding đều chậm (khoảng 160 ms), nên bị các cách nhanh hơn lấn át.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là **BoW + MLP**: từ F1 0.974 trên SUB xuống 0.324 trên 50 case khó. BoW + LogReg cũng tụt mạnh, từ 0.919 xuống 0.343. Tụt mạnh nhất ở hai nhóm: "né từ khoá" (BoW đúng 17%) và "spam giọng tự nhiên" (BoW đúng 10%).

- **Case 2 (ham): "FREE pizza in the office kitchen right now, come quick before it's gone".** Cơ chế: BoW đếm token "free", "now", "quick" và coi đây là dấu hiệu spam, nên đoán nhầm thành spam. BoW không đọc được ngữ cảnh rằng đây là tin đồng nghiệp rủ ăn pizza.
- **Case 21 (spam): "Hi Mum, I dropped my phone in the toilet… Can you send £200 to my friend's account?".** Cơ chế: không có từ khoá nào của spam, nên BoW không có tín hiệu để bắt. Embedding và LLM hiểu được ý "đóng vai người thân để xin tiền", nhưng BoW thì không.

## 5. Thử thách cá nhân

SEED = 5165, 5165 % 6 = 5, nên đề của em là **đề #6 (LLM · trả JSON)**.

Cách làm: sửa prompt để LLM trả về `{"label": ..., "reason": ...}`, viết hàm `parse_json_label`, chạy trên 10 tin SUB, so token và chi phí với bản trả về 1 từ.

Kết quả: *(điền số thật từ notebook sau khi chạy cell)*

Giải thích: *(điền sau khi có kết quả)*

## 6. Phương án cho sản phẩm thật

Yêu cầu: 10 triệu SMS/ngày (khoảng 116 tin/giây), p95 dưới 50 ms cho mỗi tin.

- **Đạt độ trễ:** BoW (p95 dưới 1 ms) và Encoder + MLP head (p95 khoảng 39 ms). Encoder + LogReg (p95 khoảng 157 ms) và LLM (khoảng 1,500 ms) không đạt.
- **Đạt cả độ trễ lẫn chất lượng:** em chọn **Encoder + MLP head (3c)**. Nó có F1 test cao nhất trong nhóm đạt SLA (0.959) và p95 khoảng 39 ms.
- **Chi phí LLM:** LLM nhỏ khoảng $0.006 mỗi 1k tin, nên 10 triệu tin/ngày khoảng $60/ngày, tức khoảng $1,800/tháng, chưa tính prompt dài hơn trong thực tế. Đây là lý do em không dùng LLM cho toàn bộ luồng.
- **Đánh đổi:** 3c cần chạy encoder MiniLM cho mỗi tin, cần dữ liệu có nhãn để train đầu phân loại, và yếu với spam giọng tự nhiên (30% trên nhóm đó). Em chấp nhận vì đây là cách cân bằng tốt nhất giữa độ trễ và chất lượng trong các thí nghiệm đã chạy.

## 7. AI log và điều đã học

- **Lần 1:** hỏi về lỗi `mat1 and mat2 must have the same dtype, but got Double and Float` trong TODO 2.2. Nhận: gợi ý ép dữ liệu về `float32`, và chỉ ra `vectorize` tạo `float64`. Sửa: dùng `np.zeros(..., dtype=np.float32)` trong `vectorize`.
- **Lần 2:** hỏi về `top_k_similar` và `cosine_similarity`. Nhận: code chuẩn hoá vector rồi nhân ma trận. Sửa: giữ cách làm này và kiểm tra bằng `top_k_similar(E_tr[0], E_tr, k=5)`.
- **Lần 3:** hỏi về `parse_label`. Nhận: kiểm tra "not spam" trước "spam". Sửa: giữ nguyên thứ tự này.
- **Lần 4:** hỏi về phần chọn đề thử thách. Nhận: đề #6 theo SEED. Sửa: tự chạy cell để xác nhận.

Điều em hiểu ra hôm nay: accuracy gây hiểu lầm khi dữ liệu lệch (luôn đoán ham vẫn đạt 87%), và điểm test sẽ bị thổi phồng nếu có dữ liệu trùng giữa train và test. Em cũng nhận ra mình hay đánh giá quá cao embedding và LLM trước khi có số liệu, nên cần chạy thử trước khi kết luận.
