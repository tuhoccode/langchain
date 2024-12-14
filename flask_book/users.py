from flask import Blueprint, render_template, request, jsonify,session,redirect,url_for,flash
from flask_book.data_user import User
import os
from flask_book.source_recomment import RecommentBook, ImageRecognition

from PIL import Image
import imagehash
recomment_instance = RecommentBook(
    embedding_file='google-bert/bert-base-uncased',
    key='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',
    csv_file='flask_book/book_genre.csv',
    repo='google/flan-t5-base',
    template="""
    Given the following text, identify and extract the title of the book. The title is typically the first line or a prominent line in the text.
    Text:
        Question: {question}
        Context: {context}
    Title:
    """
)
image_recognition = ImageRecognition()
user = Blueprint('user', __name__)


@user.route('/')
def Log():
    return redirect(url_for('user.Login'))

@user.route('/base')
def Base():
    session.permanent = False
    return render_template('base.html')

@user.route('/21 Bài Học Cho Thế Kỷ 21 (Tái Bản)')
def Bai_hoc_tk21():
    return render_template('21 Bài Học Cho Thế Kỷ 21 (Tái Bản).html')

@user.route('/48 Nguyên Tắc Chủ Chốt Của Quyền Lực')
def Ngyen_tac_48():
    return render_template('48 Nguyên Tắc Chủ Chốt Của Quyền Lực.html')

@user.route('/999 Lá Thư Gửi Cho Chính Mình (Tái Bản)')
def La_thu_999():
    return render_template('999 Lá Thư Gửi Cho Chính Mình (Tái Bản).html')

@user.route('/2666 – Roberto Bolaño (Tái Bản 2006)')
def Roberto():
    return render_template('2666 – Roberto Bolaño (Tái Bản 2006).html')

@user.route('/Bản đồ mây (Tái Bản)')
def Ban_do_may():
    return render_template('Bản đồ mây (Tái Bản).html')

@user.route('/Bí Mật Của Phan Thiên Ân (Tái Bản 2023)')
def Bi_mat_cua_thien_an():
    return render_template('Bí Mật Của Phan Thiên Ân (Tái Bản 2023).html')

@user.route('/Biến Mọi Thứ Thành Tiền - Make Money')
def Bien_moi_thu_thanh_tien():
    return render_template('Biến Mọi Thứ Thành Tiền - Make Money.html')

@user.route('/Bình Tĩnh Khi Ế Mạnh Mẽ Khi Yêu')
def Binh_tinh_khi_e():
    return render_template('Bình Tĩnh Khi Ế Mạnh Mẽ Khi Yêu.html')

@user.route('/Bộ Sách Làm Giàu Từ Chứng Khoán')
def Bo_sach_lam_giau():
    return render_template('Bộ Sách Làm Giàu Từ Chứng Khoán.html')

@user.route('/Càng Bình Tĩnh Càng Hạnh Phúc (Tái Bản)')
def Cang_binh_tinh():
    return render_template('Càng Bình Tĩnh Càng Hạnh Phúc (Tái Bản).html')

@user.route('/Càng Kỷ Luật Càng Tự Do (Tái Bản)')
def Cang_ki_luat():
    return render_template('Càng Kỷ Luật Càng Tự Do (Tái Bản).html')

@user.route('/Cây Cam Ngọt Của Tôi')
def Cay_cam_ngot():
    return render_template('Cây Cam Ngọt Của Tôi.html')

@user.route('/Cây Chuối Non Đi Giày Xanh (Bìa Mềm) - 2018')
def Cay_chuoi():
    return render_template('Cây Chuối Non Đi Giày Xanh (Bìa Mềm) - 2018.html')

@user.route('/Chiến Binh Cầu Vồng (Tái Bản 2020)')
def Chien_binh_cau_vong():return render_template('Chiến Binh Cầu Vồng (Tái Bản 2020).html')

@user.route('/Cho Tôi Xin Một Vé Đi Tuổi Thơ (Tái Bản 2023)')
def Cho_toi_xin_mot_ve_ve_tuoi_tho():
    return render_template('Cho Tôi Xin Một Vé Đi Tuổi Thơ (Tái Bản 2023).html')

@user.route('/Chuyện Con Mèo Dạy Hải Âu Bay (Tái Bản 2019)')
def Chuyen_con_meo_day_hai_au_bay():
    return render_template('Chuyện Con Mèo Dạy Hải Âu Bay (Tái Bản 2019).html')

@user.route('/Có Hai Con Mèo Ngồi Bên Cửa Sổ')
def Co_hai_con_meo_ngoi_ben_cua_so():
    return render_template('Có Hai Con Mèo Ngồi Bên Cửa Sổ.html')

@user.route('/Đắc Nhân Tâm (Tái Bản 2021)')
def Dac_nhan_tam():
    return render_template('Đắc Nhân Tâm (Tái Bản 2021).html')

@user.route('/Đàn Ông Sao Hỏa Đàn Bà Sao Kim')
def Dan_ong_sao_hoa_dan_ba_sao_kim():
    return render_template('Đàn Ông Sao Hỏa Đàn Bà Sao Kim.html')

@user.route('/Dạy Con Làm Giàu (Tái Bản) 2010')
def Day_con_lam_giau():
    return render_template('Dạy Con Làm Giàu (Tái Bản) 2010.html')

@user.route('/Đi Tìm Lẽ Sống (Tái Bản 2022-2024)')
def Di_tim_le_song():
    return render_template('Đi Tìm Lẽ Sống (Tái Bản 2022-2024).html')

@user.route('/Điều Kỳ Diệu Của Tiệm Tạp Hóa Namiya')
def Dieu_ki_dieu_cua_tiem_tam_hoa():
    return render_template('Điều Kỳ Diệu Của Tiệm Tạp Hóa Namiya.html')

@user.route('/Đứa Trẻ Hiểu Chuyện Thường Không Có Kẹo Ăn')
def Dua_tre_hieu_chuyen_thuong_khong_co_keo_an():
    return render_template('Đứa Trẻ Hiểu Chuyện Thường Không Có Kẹo Ăn.html')

@user.route('/Ghi Chép Pháp Y - Những Cái Chết Bí Ẩn')
def Ghi_chep_phap_y_nhung_cai_chet_bi_an():
    return render_template('Ghi Chép Pháp Y - Những Cái Chết Bí Ẩn.html')

@user.route('/Giận (Tái Bản 2023) (Tái Bản 2024)')
def Gian():
    return render_template('Giận (Tái Bản 2023) (Tái Bản 2024).html')

@user.route('/Giáo Trình Hán Ngữ 1(Tái Bản 2022)')
def Giao_trinh_han_ngu_1():
    return render_template('Giáo Trình Hán Ngữ 1(Tái Bản 2022).html')

@user.route('/Grammar And Vocabulary')
def Grammar_and_vocubulary():
    return render_template('Grammar And Vocabulary.html')

@user.route('/Hành Tinh Một Kẻ Nghĩ Nhiều')
def Hanh_tinh_cua_mot_ke_nghi_nhieu():
    return render_template('Hành Tinh Một Kẻ Nghĩ Nhiều.html')

@user.route('/Hiểu Về Trái Tim (Tái Bản 2023-2024)')
def Hieu_ve_trai_tim():
    return render_template('Hiểu Về Trái Tim (Tái Bản 2023-2024).html')

@user.route('/Hoàn Hảo Ngôn Ngữ Cơ Thể')
def Hoan_hao_ngon_ngu_co_the():
    return render_template('Hoàn Hảo Ngôn Ngữ Cơ Thể .html')

@user.route('/Hoàng Tử Bé (Tái Bản)')
def Hoang_tu_be():
    return render_template('Hoàng Tử Bé (Tái Bản).html')

@user.route('/Khi Hơi Thở Hóa Thinh Không (Tái Bản 2020)')
def Khi_hoi_tho_hoa_thinh_khong():
    return render_template('Khi Hơi Thở Hóa Thinh Không (Tái Bản 2020).html')

@user.route('/Không Diệt Không Sinh Đừng Sợ Hãi')
def Khong_diet_khong_sinh_dung_so_hai():
    return render_template('Không Diệt Không Sinh Đừng Sợ Hãi.html')

@user.route('/Không Phải Sói Nhưng Cũng Đừng Là Cừu')
def Khong_phai_soi_nhung_cung_dung_la_cuu():
    return render_template('Không Phải Sói Nhưng Cũng Đừng Là Cừu.html')

@user.route('/Không Sợ Chậm Chỉ Sợ Dừng (Tái Bản 2022)')
def Khong_so_cham_chi_so_dung():
    return render_template('Không Sợ Chậm Chỉ Sợ Dừng (Tái Bản 2022).html')

@user.route('/Làm Bạn Với Bầu Trời (Tái Bản 2022-2024)')
def Lam_ban_voi_bau_troi():
    return render_template('Làm Bạn Với Bầu Trời (Tái Bản 2022-2024).html')

@user.route('/Lincoln ở cõi trung ấm (Tái Bản 2022)')
def Lincoln_o_coi_trung_am():
    return render_template('Lincoln ở cõi trung ấm (Tái Bản 2022).html')

@user.route('/Lưỡng giới (Middlesex) (Tái Bản 2022)')
def Luong_gioi():
    return render_template('Lưỡng giới (Middlesex) (Tái Bản 2022).html')

@user.route('/Luyện Thi THPT Quốc Gia Môn Toán')
def Luyen_thi_thpt_quoc_gia_mon_toan():
    return render_template('Luyện Thi THPT Quốc Gia Môn Toán.html')

@user.route('/Muôn Kiếp Nhân Sinh - Tập 2')
def Muon_kiep_nhan_sinh_tap_2():
    return render_template('Muôn Kiếp Nhân Sinh - Tập 2.html')

@user.route('/Muôn Kiếp Nhân Sinh - Tập 3')
def Muon_kiep_nhan_sinh_tap_3():
    return render_template('Muôn Kiếp Nhân Sinh - Tập 3.html')

@user.route('/Muôn Kiếp Nhân Sinh')
def Muon_kiep_nhan_sinh():
    return render_template('Muôn Kiếp Nhân Sinh.html')

@user.route('/Muôn Kiếp Nhân Sinh(Bìa Cứng)')
def Muon_kiep_nhan_sinh_bia_cung():
    return render_template('Muôn Kiếp Nhân Sinh(Bìa Cứng).html')

@user.route('/Muôn Kiếp Nhân Sinh(Khổ Nhỏ)')
def Muon_kiep_nhan_sinh_kho_nho():
    return render_template('Muôn Kiếp Nhân Sinh(Khổ Nhỏ).html')

@user.route('/Ngữ Pháp Tiếng Anh Dành Cho Học Sinh')
def Ngu_phap_tieng_anh_danh_cho_hoc_sinh():
    return render_template('Ngữ Pháp Tiếng Anh Dành Cho Học Sinh.html')

@user.route('/Người bạn phi thường')
def Nguoi_ban_phi_thuong():
    return render_template('Người bạn phi thường.html')

@user.route('/Người Đua Diều (Tái Bản 2022)-2024')
def Nguoi_dua_dieu():
    return render_template('Người Đua Diều (Tái Bản 2022)-2024.html')

@user.route('/NgườI Giàu Có Nhất Babylon')
def Nguoi_giau_co_nhat_babylon():
    return render_template('NgườI Giàu Có Nhất Babylon.html')

@user.route('/Nhà Đầu Tư Thông Minh (Tái Bản 2020-2024)')
def Nha_dau_tu_thong_minh():
    return render_template('Nhà Đầu Tư Thông Minh (Tái Bản 2020-2024).html')
@user.route('/Nhà Giả Kim (Tái Bản 2020) (Tái Bản)')
def Nha_gia_kim():
    return render_template('Nhà Giả Kim (Tái Bản 2020) (Tái Bản).html')

@user.route('/Những Tù Nhân Của Địa Lý (Tái Bản 2022)')
def Nhung_tu_nhan_cua_dia_ly():
    return render_template('Những Tù Nhân Của Địa Lý (Tái Bản 2022).html')

@user.route('/Nóng Giận Là Bản Năng Tĩnh Lặng Là Bản Lĩnh')
def Nong_gian_la_ban_nang_tinh_lang_la_ban_linh():
    return render_template('Nóng Giận Là Bản Năng Tĩnh Lặng Là Bản Lĩnh.html')

@user.route('/Pachinko (Tái Bản) (Tái Bản)(Tái Bản)')
def Pachinko():
    return render_template('Pachinko (Tái Bản) (Tái Bản)(Tái Bản).html')

@user.route('/Payback Time - Ngày Đòi Nợ (Tái Bản 2022)')
def Payback_time():
    return render_template('Payback Time - Ngày Đòi Nợ (Tái Bản 2022).html')

@user.route('/Sayings Of Youth - Lời Nói Của Thanh Xuân')
def Saying_of_youth():
    return render_template('Sayings Of Youth - Lời Nói Của Thanh Xuân.html')

@user.route('/Sự Im Lặng Của Bầy Cừu (Tái Bản 2019)')
def Su_im_lang_cua_bay_cuu():
    return render_template('Sự Im Lặng Của Bầy Cừu (Tái Bản 2019).html')

@user.route('/Tâm Lý Học Tội Phạm')
def Tam_li_hoc_toi_pham():
    return render_template('Tâm Lý Học Tội Phạm.html')

@user.route('/Tâm Lý Học Về Tiền')
def Tam_li_hoc_ve_tien():
    return render_template('Tâm Lý Học Về Tiền.html')

@user.route('/Tenth of December')
def Tenth_of_december():
    return render_template('Tenth of December.html')

@user.route('/Thao Túng Tâm Lý')
def Thao_tung_tam_li():
    return render_template('Thao Túng Tâm Lý.html')

@user.route('/Thay Đổi Cuộc Sống Số Học')
def Thay_doi_cuoc_song_so_hoc():
    return render_template('Thay Đổi Cuộc Sống Số Học.html')

@user.route('/The Goldfinch (Tái Bản)')
def The_goldfinch():
    return render_template('The Goldfinch (Tái Bản).html')

@user.route('/The Savage Detectives')
def The_savage_detectives():
    return render_template('The Savage Detectives.html')

@user.route('/The Year of Magical Thinking')
def The_year_of_magical_thinking():
    return render_template('The Year of Magical Thinking.html')

@user.route('/Thiết Kế Cuộc Đời Thịnh Vượng')
def Thiet_ke_cuoc_doi_thinh_vuong():
    return render_template('Thiết Kế Cuộc Đời Thịnh Vượng.html')

@user.route('/Think And Grow Rich')
def Think_and_grow_rich():
    return render_template('Think And Grow Rich.html')

@user.route('/Thỏ Bảy Màu Và Những Người Nghĩ Nó Là Bạn')
def Tho_bay_mau_va_nhung_nguoi_nghi_no_la_ban():
    return render_template('Thỏ Bảy Màu Và Những Người Nghĩ Nó Là Bạn.html')

@user.route('/Thoát đến Phương Tây (Tái Bản)')
def Thoat_den_phuong_tay():
    return render_template('Thoát đến Phương Tây (Tái Bản).html')
@user.route('/Thuật Thao Túng (Tái Bản)')
def Thuat_thao_tung():
    return render_template('Thuật Thao Túng (Tái Bản).html')

@user.route('/Tiếng Hàn Tổng Hợp Dành Cho Người Việt Nam')
def Tieng_han_tong_hop_cho_nguoi_viet_nam():
    return render_template('Tiếng Hàn Tổng Hợp Dành Cho Người Việt Nam.html')

@user.route('/Trẻ Thông Minh Nhờ Đúng Đắn Của Cha Mẹ')
def Tre_em_thong_minh_nho_dung_dan_cua_cha_me():
    return render_template('Trẻ Thông Minh Nhờ Đúng Đắn Của Cha Mẹ.html')

@user.route('/Trí Tuệ Do Thái (Tái Bản 2022) (Tái Bản)')
def Tri_tue_do_thai():
    return render_template('Trí Tuệ Do Thái (Tái Bản 2022) (Tái Bản).html')

@user.route('/Tuổi Trẻ Đáng Giá Bao Nhiêu (Tái Bản 2021)')
def Tuoi_tre_dang_gia_bao_nhieu():
    return render_template('Tuổi Trẻ Đáng Giá Bao Nhiêu (Tái Bản 2021).html')

@user.route('/Underground Railroad')
def Underground_railroad():
    return render_template('Underground Railroad.html')

# Route xử lý upload ảnh
@user.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'Không có file hình ảnh!'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'Không có file hình ảnh!'}), 400

    # Kiểm tra và tạo thư mục 'uploads' nếu chưa tồn tại
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Lưu ảnh vào thư mục uploads
    image_path = os.path.join(upload_folder, image_file.filename)
    image_file.save(image_path)

    # So sánh ảnh với ảnh trong thư mục images
    result = recomment_instance.compare_image(image_path)
    
    # Xóa ảnh tạm thời sau khi so sánh
    os.remove(image_path)

    return jsonify({'result': result})


@user.route('/submit', methods=['POST'])
def submit():
    summary_to_search = request.form['summary']
    title = recomment_instance.find_title_summary(summary_to_search)
    return jsonify({'title': title})

@user.route('/login', methods=['POST', 'GET'])
def Login():
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email, password=password).first()
        
        if user:
            session['loggedin'] = True
            session['user_id'] = user.user_id
            session['name'] = user.name
            session['email'] = user.email
            return render_template('base.html', message=message)
        else:
            message = 'Please enter correct credentials.'
    
    return render_template('login.html', message=message)
@user.route('/logout')
def Logout():
    session.pop('loggedin', None)
    session.pop('user_id',None)
    session.pop('name',None)
    session.pop('email',None)
    return redirect(url_for('user.Login'))

@user.route('/register', methods = ['POST','GET'])
def Register():
    mesage =''
    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form and 'again_password' in request.form:
        newname = request.form['name']
        newpassword = request.form['password']
        newagainpassword = request.form['again_password']
        newemail = request.form['email']
        usered = User.query.filter_by(name = newname,email = newemail).first()

        if usered:
            mesage = 'email exited'
        elif newagainpassword != newpassword:
            mesage = 'nhap lai mat khau'
        else:
            newuser = User(name = newname,email = newemail, password=newpassword, again_password=newagainpassword)
            db.session.add(newuser)
            db.session.commit()
            mesage = 'registerd done'
            return redirect(url_for('user.Login'))

    return render_template('register.html', mesage=mesage)



