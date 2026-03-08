# DeepVAR_WeatherForecast_KLTN
Dự báo dữ liệu thời tiết đa biến bằng mô hình Deep Vector Autoregression

Sinh viên thực hiện:
- Ngô Thanh Thanh - 21110643
- Vũ Thị Bích Ngọc - 21110905
  
Nội dung báo cáo:
1.	Nghiên cứu chi tiết về bài toán dự báo thời tiết.
2.	Tìm hiểu các công trình liên quan dùng trong dự báo thời tiết.
3.	Cài đặt mô hình VAR, VAR-LSTM, DeepVAR và VAR Lai DeepVAR vào dự báo thời tiết.
4.	Thực nghiệm và so sánh hiệu năng giữa các mô hình: VAR, VAR-LSTM, DeepVAR và VAR Lai DeepVAR

DỮ LIỆU

Dữ liệu lịch sử thời tiết được lấy từ Visual Crossing Weather – nhà cung cấp hàng đầu về dữ liệu thời tiết, trong vòng 4 năm từ 18/09/2021 tới 18/09/2025 tại 4 khu vực thành phố lớn với các kiểu khí hậu khác nhau:
- Denmark – Khí hậu ôn đới hải dương
- Hồ Chí Minh – Khí hậu nhiệt đới gió mùa
- Tokyo – Khí hậu cận nhiệt đới ẩm
- St. John’s - Khí hậu cận Bắc cực hải dương
Kích thước tập dữ liệu: 1462 dòng x 34 cột

CÀI ĐẶT CÁC MÔ HÌNH
1. Mô hình VAR:
- Xác định độ trễ (lag) tối ưu để làm thông số đầu vào cho quá trình xây dựng mô hình VAR-LSTM, DeepVAR và VAR Lai DeepVAR.
2. Mô hình VAR-LSTM
- B1: Tạo dự đoán từ mô hình VAR
- B2: Tìm độ trễ tốt nhất cho dự đoán từ VAR
- B3: Chuyển đổi dự đoán từ mô hình VAR thành các cửa sổ trượt và tách X-y
- B4: Tìm siêu tham số
- B5: Xây dựng và huấn luyện mô hình LSTM trên đầu vào dự báo từ VAR
3. Mô hình DeepVAR
- B1: Tạo cửa sổ trượt từ đặc trưng trễ và tách X-y
- B2: Tìm siêu tham số
- B3: Xây dựng và huấn luyện mô hình LSTM
4. Mô hình VAR Lai DeepVAR
- B1: Tạo dự đoán từ mô hình VAR
- B2: Tính lỗi dự đoán của mô hình VAR
- B3: Chuyển đổi lỗi dự đoán từ mô hình VAR thành các cửa sổ trượt
- B4: Tìm siêu tham số
- B5: Xây dựng và huấn luyện mô hình LSTM trên đầu vào dự báo lỗi từ VAR
- Kết quả dự đoán cuối cùng là tổng hợp kết quả của dự đoán từ VAR và dự báo lỗi dự đoán từ DeepVAR

KẾT QUẢ THỰC NGHIỆM
- Đối với các TDL không dừng như Denmark, Tokyo, St John’s, cả bốn mô hình đều đạt độ chính xác dự báo cao hơn khi dữ liệu được dừng hóa. Qua đó, có thể thấy mặc dù các mô hình học sâu không yêu cầu dữ liệu phải dừng về mặt lý thuyết, việc dừng hóa dữ liệu vẫn mang lại lợi ích thực tiễn trong việc nâng cao độ chính xác và tính ổn định của dự báo.
- Các mô hình dựa trên Deep Learning, đặc biệt là DeepVAR và VAR lai DeepVAR, nhìn chung cho hiệu năng dự báo vượt trội so với mô hình VAR. Mô hình DeepVAR thể hiện tính ổn định cao và cho kết quả tốt nhất trên phần lớn các tập dữ liệu, cả trong trường hợp dữ liệu dừng và không dừng. Trong khi đó, mô hình VAR lai DeepVAR cho thấy lợi thế rõ rệt trên tập dữ liệu đã dừng sẵn (Hồ Chí Minh).
- Kết quả trên tập dữ liệu Tokyo không dừng cho thấy mô hình VAR có thể đạt hiệu năng tốt hơn các mô hình học sâu trong một số trường hợp cụ thể, vì vậy mô hình tuyến tính vẫn có giá trị khi cấu trúc dữ liệu phù hợp.
- Đối với kết quả đánh giá theo từng mô hình trên cả 4 TDL, tất cả các mô hình đều đạt hiệu năng dự báo tốt nhất trên tập dữ liệu đã dừng sẵn (Hồ Chí Minh) cho thấy dữ liệu có tính ổn định cao giúp các mô hình học được cấu trúc động một cách hiệu quả hơn. Đối với trường hợp dữ liệu không dừng, hiệu năng của các mô hình có sự khác biệt tùy theo đặc trưng khí hậu và cấu trúc dữ liệu của từng khu vực.










