# Checkpoints — Day 01

Ba mốc giúp bạn biết mình đang đúng tiến độ. Lab Coach có thể hỏi nhanh tại mỗi mốc.

## CP1 — Machine Learning chạy được (khoảng 0:40)

- **Mục tiêu:** tự viết bag-of-words + logistic regression bằng numpy.
- **Cần làm:** Phần 0 và Phần 1 của notebook (TODO 1.x), các dự đoán `predict(...)` trước mỗi thí nghiệm.
- **Sản phẩm:** các ô kiểm tra Phần 1 đều ✅; trả lời "Bắt lỗi 1–2".
- **Cần hiểu:** vì sao accuracy dễ đánh lừa khi chỉ ~13% là spam; precision và recall khác nhau ở đâu.
- **Tự kiểm tra:** giải thích được, không nhìn code, một bước cập nhật gradient descent làm gì với `w` và `b`.

## CP2 — Deep Learning và Embedding (khoảng 1:30)

- **Mục tiêu:** so được "mạng tự học đặc trưng" với "mượn encoder pre-train".
- **Cần làm:** Phần 2 (MLP + vòng lặp PyTorch) và Phần 3 (cosine, tìm kiếm theo nghĩa, kNN; hàng 3b, 3c chạy sẵn).
- **Sản phẩm:** các ô kiểm tra Phần 2–3 đều ✅; bảng kết quả có hàng 1, 2, 3a, 3b, 3c.
- **Cần hiểu:** dấu hiệu overfitting trên đường loss; vì sao embedding tổng quát hoá tốt hơn với câu "ngụy trang".
- **Tự kiểm tra:** chỉ ra được cặp hàng nào chỉ khác biểu diễn và cặp nào chỉ khác bộ phân loại.

## CP3 — LLM, tổng hợp và nộp bài (2:30)

- **Mục tiêu:** dùng LLM cho phân loại, rồi chọn phương án bằng số đo.
- **Cần làm:** Phần 4 (`call_openai`, prompt zero/few-shot, chi phí) và Phần 5 (bảng so sánh, biểu đồ Pareto, 50 case khó, tính tay, thử thách cá nhân).
- **Sản phẩm:** 4 file trong `report/individual/` theo [SUBMISSION.md](SUBMISSION.md).
- **Cần hiểu:** vì sao phải gửi `reasoning_effort="none"` và `max_completion_tokens` nhỏ; vì sao F1 trên SUB 40 tin không đủ để xếp hạng sát nút.
- **Tự kiểm tra:** với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày, nói được phương án mình chọn và lý do bằng số của chính mình.
