Hướng dẫn triển khai ứng dụng Streamlit

Bước 1: Kiểm tra và cài đặt Python

Trước tiên, bạn cần đảm bảo rằng máy đã được cài Python (đề xuất 3.11.5).

- Để xác nhận, mở terminal hoặc command prompt và chạy: python --version

Nếu chưa cài, hãy truy cập [https://www.python.org/downloads](https://www.python.org/downloads) để tải và cài đặt.

---

Bước 2: Tạo môi trường làm việc riêng (virtual environment)

Việc sử dụng môi trường ảo giúp quản lý thư viện dễ dàng và tránh xung đột giữa các dự án.

- Tạo môi trường ảo:

python -m venv env

- Kích hoạt môi trường:

	+ Trên Windows: .\env\Scripts\activate

	+ Trên macOS/Linux: source env/bin/activate

Khi kích hoạt thành công, bạn sẽ thấy tên môi trường hiển thị ở đầu dòng lệnh.

---

Bước 3: Cài đặt các gói phụ thuộc

Đảm bảo trong thư mục dự án có tệp `requirements.txt` chứa danh sách thư viện cần thiết.
Chạy lệnh sau để cài đặt:

pip install -r requirements.txt

---

Bước 4: Khởi chạy ứng dụng Streamlit

Xác nhận rằng file chính của ứng dụng có tên là `DeepVAR.py`.
Thực thi lệnh sau để chạy ứng dụng:

streamlit run DeepVAR.py

Khi lệnh chạy thành công, Streamlit sẽ mở trình duyệt web tự động. Trong trường hợp không mở, có thể truy cập thủ công bằng liên kết hiển thị trong dòng lệnh 
(thường là [http://localhost:8501]).

---

Một số lưu ý quan trọng:

Luôn đảm bảo môi trường ảo đang được kích hoạt khi cài đặt hoặc chạy ứng dụng.
Nếu gặp lỗi, hãy kiểm tra lại:

  - Python đã được cài đúng chưa?
  - Môi trường ảo có đang bật không?
  - Tệp requirements.txt có đầy đủ thư viện không?