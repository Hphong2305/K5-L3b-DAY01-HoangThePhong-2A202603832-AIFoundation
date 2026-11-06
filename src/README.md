# src/

- **File bắt đầu chạy:** `day01_spam_ml_dl_embedding_llm.ipynb` (mở bằng badge Open in Colab ở README gốc).
- **Cách chạy:** trên Colab, chạy từ trên xuống; `Runtime → Run all` khi cần chạy lại. Trên máy: `python -m pip install -r src/requirements.txt` rồi mở notebook bằng Jupyter.
- **Input:** `STUDENT_ID` (MSSV) ở ô đầu; SMS Spam Collection tự tải qua `kagglehub` (dự phòng UCI); `data/hard_eval_50.csv` (đã nhúng trong notebook); `OPENAI_API_KEY` qua Colab Secrets cho Phần 4.
- **Output:** `lab1_report.json` (tự tải về) và notebook đã chạy, nộp vào `report/individual/`.
- **File học viên được sửa:** các ô có 🧩 TODO / ✏️ trong notebook. Không sửa ô ⚙️ có sẵn, ô kiểm tra và hash `_H`. Side lab tuỳ chọn: tạo `optional_llm_api/solution/solution.py` từ `optional_llm_api/template.py`.
