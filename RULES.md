# Luật lab — Day 01

## 1. Dùng AI

- Được dùng AI để hỏi khái niệm, đọc lỗi, gợi ý cách debug.
- **Không** dán nguyên TODO cho AI viết hộ rồi nộp. Mỗi lần dùng AI ghi vào mục AI log của báo cáo: hỏi gì, nhận gì, bạn sửa gì.
- Lab Coach có thể yêu cầu giải thích hoặc sửa nhỏ bất kỳ hàm nào bạn nộp. Không giải thích được thì điểm phần code đó bị chia đôi (xem [RUBRIC.md](RUBRIC.md)).

## 2. Sao chép

- Được thảo luận ý tưởng với bạn. Code, câu trả lời và ảnh tính tay phải là của bạn.
- Mỗi người có `SEED` riêng theo MSSV, nên số liệu và đề thử thách khác nhau. Số liệu trùng y hệt bài khác là dấu hiệu sao chép.
- Bài sao chép: cả hai bài bị 0 điểm phần liên quan và báo cáo lên chương trình.

## 3. Nộp muộn

- Deadline mặc định: **23h59 (GMT+7) ngày lab**. Coach có thể thông báo deadline khác; thông báo của coach được ưu tiên.
- Commit sau deadline không được chấm, trừ khi coach đã đồng ý trước (lý do sức khoẻ, sự cố kỹ thuật có bằng chứng).

## 4. Sửa bài sau deadline

- Bản được chấm là **commit cuối cùng trước deadline**. Lab Coach chấm theo lịch sử commit, không theo trạng thái hiện tại.
- Được commit tiếp sau deadline để tự học, nhưng không làm thay đổi điểm.

## 5. Bảo mật API key và dữ liệu

- **Không bao giờ** commit API key. Trên Colab dùng 🔑 Secrets; trên máy dùng file `.env` (đã nằm trong `.gitignore`, mẫu ở `.env.example`).
- Lỡ commit key: **thu hồi (revoke) key ngay** trên trang OpenAI rồi báo coach. Xoá file thôi là chưa đủ, vì key vẫn nằm trong lịch sử git.
- Không ghi email, số điện thoại, địa chỉ vào notebook hay báo cáo. Họ tên và MSSV chỉ nằm ở tên repository và tên file báo cáo.
- Dữ liệu SMS Spam Collection dùng theo CC BY 4.0 (xem [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)); không đưa tin nhắn thật của người khác vào bài.
