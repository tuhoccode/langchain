const galleryContainer = document.querySelector('.container-item');
const galleryControlsContainer = document.querySelector('.container-controls');
const textItem = document.querySelectorAll('.itemg');
const galleryControls = ['previous', 'next'];
const galleryItems = document.querySelectorAll('.item');
const readLink = document.getElementById('readLink');

class Carousel {
    constructor(container, items, controls, itemg) {
        this.carouselContainer = container;
        this.carouselControls = controls;
        this.carouselArray = [...items];
        this.carouselItemg = [...itemg];
        this.bookUrls = [
            '/Tuổi Trẻ Đáng Giá Bao Nhiêu (Tái Bản 2021)',
            '/21 Bài Học Cho Thế Kỷ 21 (Tái Bản)',
            '/48 Nguyên Tắc Chủ Chốt Của Quyền Lực',
            '/Trẻ Thông Minh Nhờ Đúng Đắn Của Cha Mẹ',
            '/999 Lá Thư Gửi Cho Chính Mình (Tái Bản)'
        ];
    }

    updateGallery() {
        this.carouselArray.forEach(el => {
            el.classList.remove('item-1', 'item-2', 'item-3', 'item-4', 'item-5');
        });
        this.carouselArray.slice(0, 5).forEach((el, i) => {
            el.classList.add(`item-${i + 1}`);
        });
        this.updateReadLink();
    }

    updateItemg() {
        this.carouselItemg.forEach(el => {
            el.classList.remove('itemg-1', 'itemg-2', 'itemg-3', 'itemg-4', 'itemg-5');
        });
        this.carouselItemg.slice(0, 5).forEach((el, i) => {
            el.classList.add(`itemg-${i + 1}`);
        });
    }

    setCurrentState(direction) {
        if (direction.className === 'container-controls-previous') {
            this.carouselArray.unshift(this.carouselArray.pop());
            this.carouselItemg.unshift(this.carouselItemg.pop());
            this.bookUrls.unshift(this.bookUrls.pop());
        } else {
            this.carouselArray.push(this.carouselArray.shift());
            this.carouselItemg.push(this.carouselItemg.shift());
            this.bookUrls.push(this.bookUrls.shift());
        }
        this.updateGallery();
        this.updateItemg();
    }

    setControls() {
        this.carouselControls.forEach(control => {
            const button = document.createElement('button');
            button.className = `container-controls-${control}`;
            button.innerText = control;
            galleryControlsContainer.appendChild(button);
        });
    }

    useControls() {
        const triggers = [...galleryControlsContainer.childNodes];
        triggers.forEach(control => {
            control.addEventListener('click', e => {
                e.preventDefault();
                this.setCurrentState(control);
            });
        });
    }

    updateReadLink() {
        const currentBookUrl = this.getCurrentBookUrl();
        readLink.href = currentBookUrl;
    }

    getCurrentBookUrl() {
        // Luôn lấy URL của cuốn sách ở vị trí thứ 3 (giữa) trong mảng bookUrls
        return this.bookUrls[2] || '#';
    }
}

// Khởi tạo carousel
const exampleCarousel = new Carousel(galleryContainer, galleryItems, galleryControls, textItem);
exampleCarousel.updateGallery();
exampleCarousel.updateItemg();
exampleCarousel.setControls();
exampleCarousel.useControls();

// Thêm sự kiện click cho nút Đọc Sách
readLink.addEventListener('click', (e) => {
    e.preventDefault();
    // chặn tính năng của thẻ hiện tại trong trường hợp là thẻ a    
    const currentUrl = exampleCarousel.getCurrentBookUrl();
    if (currentUrl !== '#') {
        window.location.href = currentUrl;
    }
});