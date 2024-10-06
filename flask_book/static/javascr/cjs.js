const containerItem = document.querySelector('.container-item');
const item = document.querySelectorAll('.item');
const itemg = document.querySelectorAll('.itemg');
const control = ['previous', 'next'];
const containerControl = document.querySelector('.container-controls');
const RLink = document.getElementById('readLink');

class Carousel {
    constructor(container, item, itemg, control) {
        this.carouselContainer = container;
        this.carouselItem = [...item];
        this.carouselItemg = [...itemg];
        this.carouselControl = control;
        this.bookUrl = [
            // hang 1
            '/21 Bài Học Cho Thế Kỷ 21 (Tái Bản)',
            '/48 Nguyên Tắc Chủ Chốt Của Quyền Lực',
            '/21 Bài Học Cho Thế Kỷ 21 (Tái Bản)',
            '/Trẻ Thông Minh Nhờ Đúng Đắn Của Cha Mẹ',
            '/999 Lá Thư Gửi Cho Chính Mình (Tái Bản)',
            '/2666 – Roberto Bolaño (Tái Bản 2006)',
            // hang 2
            '/Biến Mọi Thứ Thành Tiền - Make Money',
            '/Bí Mật Của Phan Thiên Ân (Tái Bản 2023)',
            '/Bình Tĩnh Khi Ế Mạnh Mẽ Khi Yêu',
            '/Luyện Thi THPT Quốc Gia Môn Toán',
            '/Bộ Sách Làm Giàu Từ Chứng Khoán',
            '/Càng Bình Tĩnh Càng Hạnh Phúc (Tái Bản)',
            // hang 3
            '/Càng Kỷ Luật Càng Tự Do (Tái Bản)',
            '/Cây Cam Ngọt Của Tôi',
            '/Cây Chuối Non Đi Giày Xanh (Bìa Mềm) - 2018',
            '/Chiến Binh Cầu Vồng (Tái Bản 2020)',
            '/Cho Tôi Xin Một Vé Đi Tuổi Thơ (Tái Bản 2023)',
            '/Chuyện Con Mèo Dạy Hải Âu Bay (Tái Bản 2019)',
            // hang 4
            '/Bản đồ mây (Tái Bản)',
            '/Có Hai Con Mèo Ngồi Bên Cửa Sổ',
            '/Grammar And Vocabulary',
            '/Hoàn Hảo Ngôn Ngữ Cơ Thể ',
            '/Đắc Nhân Tâm (Tái Bản 2021)',
            '/Đàn Ông Sao Hỏa Đàn Bà Sao Kim',
            // hang 5
            '/Dạy Con Làm Giàu (Tái Bản) 2010',
            '/Điều Kỳ Diệu Của Tiệm Tạp Hóa Namiya',
            '/Đi Tìm Lẽ Sống (Tái Bản 2022-2024)',
            '/Đứa Trẻ Hiểu Chuyện Thường Không Có Kẹo Ăn',
            '/Thoát đến Phương Tây (Tái Bản)',
            '/Ghi Chép Pháp Y - Những Cái Chết Bí Ẩn',
            // hang 6
            '/Giận (Tái Bản 2023) (Tái Bản 2024)',
            '/Giáo Trình Hán Ngữ 1(Tái Bản 2022)',
            '/Hành Tinh Một Kẻ Nghĩ Nhiều',
            '/Hiểu Về Trái Tim (Tái Bản 2023-2024)',
            '/Hoàng Tử Bé (Tái Bản)',
            '/Khi Hơi Thở Hóa Thinh Không (Tái Bản 2020)',
            // hang 7
            '/Không Diệt Không Sinh Đừng Sợ Hãi',
            '/Không Phải Sói Nhưng Cũng Đừng Là Cừu',
            '/Không Sợ Chậm Chỉ Sợ Dừng (Tái Bản 2022)',
            '/Làm Bạn Với Bầu Trời (Tái Bản 2022-2024)',
            '/Lincoln ở cõi trung ấm (Tái Bản 2022)',
            '/Lưỡng giới (Middlesex) (Tái Bản 2022)',
            // hang 8
            '/Muôn Kiếp Nhân Sinh',
            '/Muôn Kiếp Nhân Sinh(Bìa Cứng)',
            '/Muôn Kiếp Nhân Sinh(Khổ Nhỏ)',
            '/Muôn Kiếp Nhân Sinh - Tập 2',
            '/Muôn Kiếp Nhân Sinh - Tập 3',
            '/Người bạn phi thường',
            // hang 9
            '/Người Đua Diều (Tái Bản 2022)-2024',
            '/NgườI Giàu Có Nhất Babylon',
            '/Ngữ Pháp Tiếng Anh Dành Cho Học Sinh',
            '/Nhà Đầu Tư Thông Minh (Tái Bản 2020-2024)',
            '/Nhà Giả Kim (Tái Bản 2020) (Tái Bản)',
            '/Những Tù Nhân Của Địa Lý (Tái Bản 2022)',
            // hang 10
            '/Nóng Giận Là Bản Năng Tĩnh Lặng Là Bản Lĩnh',
            '/Nóng Giận Là Bản Năng Tĩnh Lặng Là Bản Lĩnh',
            '/Pachinko (Tái Bản) (Tái Bản)(Tái Bản)',
            '/Payback Time - Ngày Đòi Nợ (Tái Bản 2022)',
            '/Sayings Of Youth - Lời Nói Của Thanh Xuân',
            '/Sự Im Lặng Của Bầy Cừu (Tái Bản 2019)',
            // hang 11
            '/Tâm Lý Học Tội Phạm',
            '/Tâm Lý Học Về Tiền',
            '/Tenth of December',
            '/Thao Túng Tâm Lý',
            '/Thay Đổi Cuộc Sống Số Học',
            '/The Goldfinch (Tái Bản)',
            // hang 12
            '/The Savage Detectives',
            '/Underground Railroad',
            '/The Year of Magical Thinking',
            '/Thiết Kế Cuộc Đời Thịnh Vượng',
            '/Thiết Kế Cuộc Đời Thịnh Vượng',
            '/Think And Grow Rich',
            // hang 13
            '/Thỏ Bảy Màu Và Những Người Nghĩ Nó Là Bạn',
            '/Thuật Thao Túng (Tái Bản)',
            '/Tiếng Hàn Tổng Hợp Dành Cho Người Việt Nam',
            '/Trí Tuệ Do Thái (Tái Bản 2022) (Tái Bản)',
            '/Trí Tuệ Do Thái (Tái Bản 2022) (Tái Bản)',
            '/Trí Tuệ Do Thái (Tái Bản 2022) (Tái Bản)',
        ,];
        this.currentIndex = 0;
    }

    updateItem() {
        this.carouselItem.forEach(el => {
            el.classList.remove(...Array.from({ length: 999}, (_, i) => `item-${i + 1}`));
        });
        const countItem = Math.min(999, this.carouselItem.length);
        this.carouselItem.slice(0, countItem).forEach((el, i) => {
            el.classList.add(`item-${i + 1}`);
        });
        this.updateReadLink();
    }

    updateItemg() {
        this.carouselItemg.forEach(el => {
            el.classList.remove(...Array.from({ length: 999 }, (_, i) => `itemg-${i + 1}`));
        });
        const countItemg = Math.min(999, this.carouselItemg.length);
        this.carouselItemg.slice(0, countItemg).forEach((el, i) => {
            el.classList.add(`itemg-${i + 1}`);
        });
    }

    setNextState() {
        this.carouselItem.push(this.carouselItem.shift());
        this.carouselItemg.push(this.carouselItemg.shift());
        this.bookUrl.push(this.bookUrl.shift());
        this.currentIndex = 1; 
        this.updateItem();
        this.updateItemg();
        this.updateReadLink();
    }

    setPreviousState() {
        this.carouselItem.unshift(this.carouselItem.pop());
        this.carouselItemg.unshift(this.carouselItemg.pop());
        this.bookUrl.unshift(this.bookUrl.pop());
        this.currentIndex = 0;
        this.updateItem();
        this.updateItemg();
        this.updateReadLink();
    }

    setControl() {
        this.carouselControl.forEach(control => {
            const button = document.createElement('button');
            button.className = `container-controls-${control}`;
            button.innerText = control;
            containerControl.appendChild(button);
        });
    }

    useControl() {
        const triggers = [...containerControl.childNodes];
        triggers.forEach(control => {
            control.addEventListener('click', e => {
                e.preventDefault();
                if (control.classList.contains('container-controls-next')) {
                    this.setNextState();
                } else if (control.classList.contains('container-controls-previous')) {
                    this.setPreviousState();
                }
            });
        });
    }

    updateReadLink() {
        const currentLink = this.getCurrentLink();
        RLink.href = currentLink;
    }

    getCurrentLink() {
        return this.bookUrl[this.currentIndex + 1] || '/Tuổi Trẻ Đáng Giá Bao Nhiêu (Tái Bản 2021)';
    }

    autoslide() {
        setInterval(() => {
            this.setNextState();
        }, 3000);
    }
}

const runjs = new Carousel(containerItem, item, itemg, control);
runjs.updateItem();
runjs.updateItemg();
runjs.setControl();
runjs.useControl();
runjs.autoslide();

RLink.addEventListener('click', e => {
    e.preventDefault();
    const currentUrl = runjs.getCurrentLink();
    if (currentUrl !== '#') {
        window.location.href = currentUrl;
    }
});