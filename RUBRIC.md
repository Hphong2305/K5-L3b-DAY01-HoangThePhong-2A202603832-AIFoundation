# Rubric — Lab Ngày 1: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> **Cách chấm:** mở **link repository** học viên nộp trên LMS, đọc thư mục `report/individual/`
> (`MSSV_HoTen.md`, `day01_lab.ipynb`, `lab1_report.json`, `hand_calc.jpg`). Chạy lại notebook với MSSV trong tên repository
> sẽ cho đúng `SEED` và cùng cách chia dữ liệu như bài của học viên.

| Hạng mục | Điểm | Đủ bằng chứng | Chưa đạt |
| --- | ---: | --- | --- |
| Bộ kiểm tra tự động (TODO 1.1 → 4.4) | 35 | Các ô kiểm tra ✅ trong notebook đã chạy và trong `lab1_report.json` | ❌ / ⬜, hoặc hàm dùng `sklearn` trong vùng tự code |
| Dự đoán + câu hỏi + giải thích (Phần 0–4) | 25 | Đủ 10 dự đoán có dấu thời gian **trước** thí nghiệm; câu trả lời dẫn số liệu của chính mình và giải thích đúng cơ chế | Thiếu dự đoán, trả lời chung chung không gắn với kết quả chạy |
| Bắt lỗi 1–3 | 10 | Chỉ ra đúng lỗi (data leakage / đánh giá trên train, thiếu chuẩn hoá + sai chiều sắp xếp, `"not spam"` chứa `"spam"`) và hậu quả | Nêu lỗi cú pháp/đặt tên thay vì lỗi phương pháp |
| Tính tay | 10 | Ảnh viết tay có các bước tính (phần chính của điểm) + 5 giá trị khớp hash | Chỉ có số, không có ảnh hoặc ảnh không có bước tính. Hash chỉ là bước tự kiểm tra, dò ngược được, nên không đủ để có điểm |
| Thử thách cá nhân + phản ánh + AI log | 20 | Đúng đề theo `SEED`, có code + kết quả + giải thích; phản ánh dùng bảng so sánh của mình; AI log trung thực | Làm sai đề, không có kết quả, phản ánh không dẫn số liệu |
| **Tổng điểm chính** | **100** | | |

## Bonus (tách riêng, tối đa 10 điểm)

Bonus là **điểm cộng của lab**, không phải điểm chuyên cần, và không bù cho hạng mục chính bị thiếu bằng chứng.

| Hạng mục bonus | Tối đa | Đủ bằng chứng |
| --- | ---: | --- |
| Case khó tự nghĩ (`MY_CASES`, Phần 5) | 5 | Ít nhất 5 case gốc, mỗi case có `why_tricky`, kèm nhận xét model nào bị lừa và vì sao |
| Side lab LLM API tuỳ chọn (`src/optional_llm_api/`) | 5 | `src/optional_llm_api/solution/solution.py` chạy qua `pytest tests/test_solution.py -v`, có trả lời bài tập trong `docs/optional-llm-api-exercises.md` |

## Trừ điểm và điều kiện không chấm

| Tình huống | Xử lý |
| --- | --- |
| Lộ API key trong notebook, report hoặc lịch sử commit | −10 điểm; học viên phải thu hồi (revoke) key ngay. Xoá file không đủ, vì key vẫn còn trong lịch sử git |
| Email, số điện thoại hoặc dữ liệu cá nhân khác trong bài | −5 điểm, yêu cầu gỡ |
| Sai tên repository hoặc sai tên file báo cáo (xem `SUBMISSION.md`) | −5 điểm nếu vẫn tìm được bài; không chấm nếu không tìm được |
| Nộp muộn hoặc sửa sau deadline | Theo `RULES.md` |
| Sao chép bài hoặc dùng AI không khai báo | Theo `RULES.md` |

## Quy tắc vấn đáp

Lab Coach có thể chọn ngẫu nhiên một hàm học viên đã nộp và yêu cầu giải thích, hoặc sửa nhỏ tại chỗ (ví dụ:
"đổi dấu bước cập nhật thì loss thế nào?"). Nếu không giải thích được, điểm phần code tương ứng bị **chia đôi**.

## Evidence tối thiểu

- `report/individual/MSSV_HoTen.md` điền đủ các mục, không để trống.
- `report/individual/day01_lab.ipynb` đã chạy (có output), không chứa API key.
- `report/individual/lab1_report.json` có `seed`, `checks`, `predictions`, `results`.
- `report/individual/hand_calc.jpg` là ảnh viết tay.
- Không có email, số điện thoại, API key trong notebook hoặc report. Họ tên chỉ nằm ở tên repository và tên file báo cáo.
