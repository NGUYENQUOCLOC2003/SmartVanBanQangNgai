# SmartVanBanQangNgai

## Giới thiệu

Dự án SmartVanBanQangNgai là một hệ thống chatbot thông minh, được xây dựng bằng Python và Streamlit, giúp người dùng quản lý và tìm kiếm thông tin văn bản một cách hiệu quả.

## Yêu cầu hệ thống

* Python 3.10 (Bắt buộc)
* Hệ điều hành: Windows, MacOS, hoặc Linux

### Cài đặt Python 3.10

* Tải Python 3.10 từ liên kết sau: [Python 3.10.17 Download](https://www.python.org/downloads/release/python-31017/)
* Đảm bảo đã chọn tùy chọn "Add Python to PATH" trong quá trình cài đặt.

## Hướng dẫn cài đặt thư viện

Cài đặt thư viện yêu cầu:

```bash
py -3.10 -m pip install -r requirements_local.txt
```

## Hướng dẫn chạy ứng dụng

### 1. Chạy Admin

* Để khởi động giao diện quản trị (Admin), chạy lệnh sau:

```bash
py -3.10 -m streamlit run admin_chatbot.py
```

### 2. Chạy User

* Để khởi động giao diện người dùng (User), chạy lệnh sau:

```bash
py -3.10 -m streamlit run botchat_user.py
```

## Lưu ý

* Phải đảm bảo đã cài đặt Python 3.10 trước khi tiếp tục.
* Đường dẫn Python có thể khác tùy vào hệ điều hành của bạn.
* Đảm bảo file `requirements_local.txt` có trong thư mục dự án.
