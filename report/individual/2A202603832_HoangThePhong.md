# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**
- **Họ và tên:**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | x |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | x |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | x |

## 2. Kết quả chính

Bảng so sánh 4 thế hệ AI trên cùng bài toán lọc SMS Spam (`SEED = 3886`):

| Phương pháp | Precision | Recall | F1 (test) | F1 (SUB 40) | train (s) | tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
|---|---|---|---|---|---|---|---|---|---|
| **1. BoW + LogReg (tự code)** | 0.989 | 0.822 | **0.898** | **0.889** | 1.425 | 23,309.774 | 0.068 | **0.248** | **0.000** |
| **2. BoW + MLP (deep learning)** | 1.000 | 0.869 | **0.930** | **0.947** | 16.887 | 20,430.407 | 0.330 | **0.587** | **0.000** |
| **3a. Encoder (MiniLM) + kNN** | 0.891 | 0.916 | **0.903** | **0.895** | 102.844 | 33.279 | 52.222 | **66.282** | **0.000** |
| **3b. Encoder (MiniLM) + LogReg** | 0.953 | 0.953 | **0.953** | **0.950** | 104.762 | 42.542 | 30.090 | **41.580** | **0.000** |
| **3c. Encoder (MiniLM) + MLP head** | 0.953 | 0.944 | **0.948** | **0.950** | 103.929 | 44.892 | 31.671 | **42.191** | **0.000** |
| **4a. LLM nhỏ zero-shot (`gemini-3.5-flash-lite`)** | 1.000 | 0.950 | — | **0.974** | 0.000 | 1.812 | 610.393 | **772.047** | **0.023** |
| **4b. LLM lớn zero-shot (`gemini-3.8-flash`)** | 1.000 | 1.000 | — | **1.000** | 0.000 | 8.970 | 111.125 | **111.535** | **0.053** |
| **4c. LLM nhỏ few-shot (4 ví dụ)** | 1.000 | 1.000 | — | **1.000** | 0.000 | 12.984 | 76.132 | **76.567** | **0.059** |

- **Trên đường Pareto:** `1. BoW + LogReg (tự code)`, `2. BoW + MLP (deep learning)`, `3b. Encoder (MiniLM) + LogReg`, `4c. LLM nhỏ few-shot`.
- **Đạt SLA (`p95 < 50 ms`):** `1. BoW + LogReg`, `2. BoW + MLP`, `3b. Encoder (MiniLM) + LogReg`, `3c. Encoder (MiniLM) + MLP head`.

---

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có sự khác biệt đáng chú ý nhất so với thực tế chạy thực nghiệm:

1. **`dl_curve` (So sánh BoW + MLP với BoW + LogReg):**
   - **Dự đoán:** F1 của MLP sẽ chỉ tương đương hoặc nhỉnh hơn không đáng kể so với Logistic Regression (~0.90) do đầu vào vẫn là Bag-of-Words và tập dữ liệu nhỏ.
   - **Thực tế:** Trên tập test đầy đủ, `2. BoW + MLP` đạt `F1 = 0.930` (Precision tuyệt đối `1.000`, Recall `0.869`) và trên tập `SUB 40` đạt `0.947`, tăng khá rõ so với `0.898` (test) và `0.889` (SUB) của `1. BoW + LogReg`.
   - **Vì sao lệch:** Lớp ẩn 64 nơ-ron kết hợp hàm kích hoạt phi tuyến `ReLU` + thuật toán tối ưu `Adam` học được các tổ hợp đồng xuất hiện của nhiều từ (ví dụ `"call"` + `"now"` + ký hiệu tiền tệ) tốt hơn so với mô hình cộng tuyến tính đơn thuần với `lr` cố định.

2. **`tradeoff` (Độ trễ p95 của nhóm Embedding 3a vs 3b/3c so với ngưỡng SLA 50 ms):**
   - **Dự đoán:** Mọi phương pháp dùng chung encoder `MiniLM` (3a, 3b, 3c) sẽ đều nằm bên trái vạch SLA `< 50 ms` vì bước tốn thời gian nhất là chạy encoder.
   - **Thực tế:** `3b. Encoder + LogReg` (`p95 = 41.580 ms`) và `3c. Encoder + MLP head` (`p95 = 42.191 ms`) đạt SLA, nhưng **`3a. Encoder + kNN` bị vượt SLA (`p95 = 66.282 ms`)** và bị lấn át hoàn toàn trên biểu đồ Pareto.
   - **Vì sao lệch:** Khi gọi từng tin trên CPU của Colab, ngoài thời gian encode câu truy vấn (~30 ms), `kNN` còn phải chuẩn hoá và nhân ma trận tìm kiếm trên toàn bộ ~3.600 vector của tập train ở mỗi lần gọi, cộng thêm biến thiên độ trễ trên CPU khiến `p95` vượt mốc 50 ms.

3. **`hard_eval` (Mức độ sụt giảm của BoW trên bộ case khó):**
   - **Dự đoán:** Accuracy và F1 của các mô hình truyền thống (BoW + LogReg, BoW + MLP) trên bộ case khó sẽ giảm xuống khoảng `50% – 65%`.
   - **Thực tế:** Accuracy đúng là ở mức `50.9%` (LogReg) và `54.5%` (MLP), nhưng **chỉ số F1 của `1. BoW + LogReg` tụt thê thảm từ `0.889` xuống chỉ còn `0.308`** (và `2. BoW + MLP` tụt từ `0.947` xuống `0.390`).
   - **Vì sao lệch:** Do dữ liệu huấn luyện gốc mất cân bằng (~87% là `ham`), bias `b` học được mang giá trị âm. Khi gặp các tin spam viết giọng tự nhiên (`natural_spam`), giả danh (`manipulation`) hoặc viết biến dạng ký tự (`obfuscation`), không có từ khoá spam nào khớp với từ điển 2.000 từ nên mô hình đẩy hầu hết về `ham` (Recall cực thấp, làm F1 sụp đổ).

---

## 4. Phân tích 50 case khó

Bảng so sánh F1 trên tập `SUB 40` thông thường và bộ case khó (50 case chuẩn + 5 case tự thêm):

| Mô hình | F1 test thường (SUB) | F1 bộ case khó | Accuracy bộ case khó | Độ tụt F1 |
|---|---|---|---|---|
| **1. BoW + LogReg** | 0.889 | **0.308** | 0.509 | **−0.581** |
| **2. BoW + MLP** | 0.947 | **0.390** | 0.545 | **−0.557** |
| **3a. Emb + kNN** | 0.895 | **0.776** | 0.727 | −0.119 |
| **3b. Emb + LogReg** | 0.950 | **0.655** | 0.655 | −0.295 |
| **3c. Emb + MLP head** | 0.950 | **0.655** | 0.655 | −0.295 |
| **4a. LLM nhỏ zero-shot** | 0.974 | **0.897** | 0.891 | −0.077 |
| **4d. LLM nhỏ + prompt phòng thủ** | 0.974 | **0.984** | 0.982 | **+0.010** |

- **Model tụt nhiều nhất:** `1. BoW + LogReg` (tụt `0.581` điểm F1, từ `0.889` xuống `0.308`) và `2. BoW + MLP` (tụt `0.557` điểm F1). Các nhóm làm BoW sai gần như toàn bộ là **Né từ khoá (`obfuscation`)**, **Spam giọng tự nhiên (`natural_spam`)**, **Thao túng/lừa đảo (`manipulation`)**, và **Nói về spam (`talks_about_spam`)**.
- **Phân tích 2 case cụ thể và cơ chế gây lỗi:**
  1. **Case #30 (`obfuscation`, nhãn thật = `spam`):**
     > *"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"*
     - **Cơ chế sai ở BoW (`1`, `2`):** Hàm `tokenize` tách chữ theo regex `[a-z0-9£$€]+`. Khi kẻ tấn công chèn dấu cách vào giữa từng chữ cái (`"F R E E"`, `"t x t"`), bộ tách từ biến chúng thành các token đơn lẻ `["f", "r", "e", "e", "t", "x", "t"]`. Các từ khoá có trọng số dương lớn như `"free"`, `"entry"`, `"txt"` hoàn toàn biến mất khỏi vector Bag-of-Words (`X`), khiến tích vô hướng `w·x + b < 0` và mô hình đoán nhầm thành `ham`.
  2. **Case #15 (`natural_spam`, nhãn thật = `spam` — case duy nhất cả 7 mô hình đều sai):**
     > *"Good morning! Your neighbour recommended our cleaning service, first clean is on us this week, reply with a time that suits"*
     - **Cơ chế sai ở cả BoW, Embedding và LLM:** Câu này không dùng bất kỳ từ khoá khuyến mãi lộ liễu nào (`"first clean is on us"` thay cho `"free"`), lại mở đầu bằng lời chào lịch sự và nhắc đến `"neighbour"` (hàng xóm). Trong không gian BoW và Embedding MiniLM, câu này nằm sát cụm tin nhắn giao tiếp đời thường (`ham`); ngay cả LLM cũng bị đánh lừa vì ngữ điệu giống hệt một dịch vụ địa phương do người quen giới thiệu.

---

## 5. Thử thách cá nhân

- **Đề bài theo `SEED = 3886` (`MY_CHALLENGE = 3886 % 6 = 4` → Đề #5):**
  > *"Embedding · Viết 4 tin nhắn của riêng bạn (2 spam né từ khoá, 2 ham chứa 'free'/'win'). So dự đoán BoW vs Embedding+LogReg và giải thích từng tin."*

- **Cách làm:**
  Tự thiết kế 4 tin nhắn tiếng Anh:
  - 2 tin `spam` cố tình không dùng từ `"free"`, `"win"`, `"prize"` mà diễn đạt bằng *"at zero charge"* (miễn phí) và kịch bản lừa đảo xác thực thẻ ngân hàng (*"confirm your PIN"*).
  - 2 tin `ham` là hội thoại sinh viên đời thường nhưng chứa dày đặc các từ `"free"`, `"win"`, `"won"`, `"vouchers"`.
  Đưa cả 4 tin qua `1. BoW + LogReg` và `3b. Encoder (MiniLM) + LogReg` (đã chuẩn hoá theo `mu, sd`) để đối chiếu nhãn dự đoán.

- **Kết quả chạy thực tế:**
  1. `[Thật = spam]` *"Hi neighbour, we are clearing out stock—get a luxury mattress delivered at zero charge if you reply YES today"*
     - `BoW + LogReg`: đoán **`ham` (Sai ✗)** · `Encoder + LogReg`: đoán **`spam` (Đúng ✓)**
     - **Giải thích:** Cụm `"at zero charge"` không khớp với cột `"free"` trong từ điển BoW, còn từ `"neighbour"` và `"today"` kéo trọng số về phía `ham`. Ngược lại, encoder MiniLM hiểu nghĩa cả câu là chào hàng khuyến mãi và yêu cầu `"reply YES"` nên phân loại đúng là `spam`.
  2. `[Thật = spam]` *"Your bank card security check failed, confirm your PIN at barclays-auth-check.org within 2 hours"*
     - `BoW + LogReg`: đoán **`ham` (Sai ✗)** · `Encoder + LogReg`: đoán **`spam` (Đúng ✓)**
     - **Giải thích:** Các từ `"bank"`, `"security"`, `"check"`, `"pin"` hiếm xuất hiện trong tập SMS spam quảng cáo cổ điển nên BoW không gắn trọng số dương đủ lớn. Trong khi đó, biểu diễn ngữ nghĩa của MiniLM nhận diện được cấu trúc khẩn cấp kèm đường link yêu cầu xác thực là hành vi đáng ngờ (`spam`).
  3. `[Thật = ham]` *"Are you free after class? We can watch the team win the finals together!"*
     - `BoW + LogReg`: đoán **`ham` (Đúng ✓)** · `Encoder + LogReg`: đoán **`ham` (Đúng ✓)**
     - **Giải thích:** Dù có cả `"free"` và `"win"`, các từ ngữ cảnh xung quanh như `"class"`, `"watch"`, `"together"` có trọng số âm (đặc trưng của `ham`) đủ bù lại trong BoW, và hoàn toàn rõ nghĩa hội thoại bạn bè trong không gian Embedding.
  4. `[Thật = ham]` *"I won two free vouchers for coffee at the campus library quiz!"*
     - `BoW + LogReg`: đoán **`spam` (Sai ✗)** · `Encoder + LogReg`: đoán **`ham` (Đúng ✓)**
     - **Giải thích:** Ba từ `"won"`, `"free"`, `"vouchers"` xuất hiện cùng lúc tạo tổng trọng số dương rất lớn trong BoW khiến mô hình lập tức gắn nhãn `spam`, bất chấp ngữ cảnh vô hại là `"coffee at the campus library"`. Mô hình Embedding hiểu đây là câu khoe với bạn bè ở thư viện trường nên đoán đúng `ham`.

---

## 6. Phương án cho sản phẩm thật

**Bài toán:** Nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình, giờ cao điểm có thể gấp 5–10 lần), yêu cầu độ trễ **p95 < 50 ms/tin**.

1. **Lựa chọn phương án chính:** Chọn **`3b. Encoder (MiniLM) + LogReg`** (kết hợp kiến trúc phân tầng **Cascading Router** với `1. BoW + LogReg`).
2. **Dẫn chứng số liệu từ bảng thực nghiệm:**
   - **Về độ trễ (SLA):**
     - Các phương án LLM (`4a`, `4b`, `4c`) có `p95` từ **`76.57 ms` đến `772.05 ms`**, đều **vi phạm SLA `< 50 ms`**, và thông lượng tuần tự chỉ đạt `1.8 – 13 tin/s` (không đáp ứng nổi tải trung bình `116 tin/s`).
     - Hơn nữa, nếu dùng LLM cho 10 triệu tin/ngày: với mức giá thực nghiệm `$0.023 / 1k tin` (`4a`) đến `$0.053 / 1k tin` (`4b`), chi phí sẽ tốn **`$230 – $530 / ngày` (~`$6,900 – $15,900 / tháng`)**; ngay cả ở mức tối thiểu lý thuyết `$10.50 / 1M tin` (`HAND_CALC`) cũng tốn `$3,150 / tháng`.
     - Trong các mô hình đạt SLA (`p95 < 50 ms`) và chi phí API = **$0.00**, **`3b. Encoder (MiniLM) + LogReg`** đạt **F1 cao nhất trên tập test (`0.953`)**, cân bằng hoàn hảo giữa Precision (`0.953`) và Recall (`0.953`), với `p95 = 41.580 ms < 50 ms`.
   - **Tối ưu hoá thông lượng bằng Cascading Router:**
     - Đặt **`1. BoW + LogReg`** (`p95 = 0.248 ms`, thông lượng `23,310 tin/s`) kèm bước chuẩn hoá chữ viết biến dạng (de-obfuscation) làm **Tầng 1**: xử lý ngay ~80% tin nhắn cực kỳ rõ ràng ($p < 0.03$ hoặc $p > 0.97$).
     - Chỉ ~20% tin nhắn còn lại mới đi qua **Tầng 2 (`3b. Encoder MiniLM + LogReg`, `p95 = 41.58 ms`)**. Nhờ đó độ trễ trung bình toàn hệ thống giảm xuống dưới `10 ms`, tải GPU/CPU giảm 5 lần.
     - **Tầng 3 (`4d. LLM + Prompt phòng thủ`, `F1 case khó = 0.984`)** chỉ chạy **bất đồng bộ (offline/background)** trên < 0.5% mẫu nghi ngờ ($0.4 \le p \le 0.6$) để phát hiện chiến dịch lừa đảo (phishing/manipulation) mới và tự động bổ sung nhãn huấn luyện lại cho Tầng 1 & 2 mỗi ngày.
3. **Đánh đổi (Trade-offs) chấp nhận:**
   - Chấp nhận thời gian khởi tạo/huấn luyện ban đầu lâu hơn BoW (`104.8s` để encode tập train so với `1.4s` của BoW) và tốn bộ nhớ RAM (~90MB cho trọng số MiniLM) để đổi lấy khả năng hiểu ngữ nghĩa vượt trội (`F1 50 case khó` đạt `0.655` so với `0.308` của BoW).
   - Ở đường truyền thời gian thực (real-time inline), chấp nhận việc Tầng 2 có thể bỏ lọt một số mẫu lừa đảo giả danh cực kỳ tinh vi chưa từng gặp (như kịch bản *"Hi Mum"* hay *"neighbour cleaning service"*), nhưng bù lại đảm bảo tuyệt đối SLA `< 50 ms` và không chặn nhầm tin OTP/giao dịch hợp lệ của khách hàng.

---

## 7. AI log và điều đã học

### 🤖 AI log
| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| Phần 4 (4.1) | Cách khắc phục lỗi `RateLimitError: 429 insufficient_quota` khi hết credit OpenAI và chuyển sang dùng Google Gemini API tương thích với `call_openai` | Hướng dẫn đổi `LLM_PROVIDER = "gemini"`, dùng endpoint OpenAI-compatible của Google và truyền `**provider_kwargs(max_tokens)` trong `call_openai` | Phát hiện lỗi `404` với `gemini-2.5-flash` và lỗi `400 Thinking level MINIMAL is not supported` trên `gemini-3.8-flash`; đã đổi sang `"reasoning_effort": "low"` và kết hợp `gemini-3.8-flash` / `gemini-3.5-flash-lite`, kiểm tra pass ô `#5451ccaa` và check `27c60c01` (`✅`) |
| Phần 4 (4.2 – 4.4) | Viết hàm `build_classification_prompt`, `parse_label` và `llm_classify_batch` theo docstring của bài lab | Cung cấp code ghép few-shot examples, chuẩn hoá chuỗi trả về để lấy nhãn `0`/`1`/`None` và tính chi phí token ước lượng `số_từ / 0.75` | Chạy unit test `_run_check("4.2–4.3 prompt / parse_label", _t)` và `_run_check("4.1 + 4.4 call_openai / llm_classify_batch", _t)` đạt `✅`, kiểm chứng đầu ra thực tế trên 40 tin `SUB` và bộ 50 case khó |

### 💡 Điều đã học hôm nay
Trước buổi lab, mình nghĩ mô hình càng lớn và hiện đại (như Deep Learning hay LLM) thì lúc nào cũng vượt trội hoàn toàn so với Machine Learning cổ điển. Hôm nay tự tay cài đặt và đo đạc cả 4 thế hệ AI trên cùng một bài toán, mình hiểu ra rằng trên dữ liệu chuẩn cùng phân phối, `BoW + LogReg` chỉ vài nghìn phép nhân đã đạt `F1 = 0.898` với tốc độ `0.25 ms/tin`, còn `MiniLM + LogReg` (`F1 = 0.953`, `p95 = 41.6 ms`) mới là điểm cân bằng tối ưu cho sản phẩm thực tế có ràng buộc SLA `< 50 ms`. LLM kèm prompt phòng thủ (`<sms>...</sms>`) tuy quá chậm và đắt để lọc trực tiếp hàng chục triệu tin nhắn mỗi ngày, nhưng lại cực kỳ mạnh khi xử lý các mẫu tấn công đối kháng (obfuscation, social engineering, prompt injection) và giải quyết bài toán *cold-start* khi chưa có dữ liệu gán nhãn.gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
