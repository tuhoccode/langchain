const containerItem = document.querySelector('.container-item');
const item = document.querySelectorAll('.item');
const itemg = document.querySelectorAll('.itemg');
const control = ['previous', 'next'];
const containerControl = document.querySelector('.container-controls');
const readLink = document.getElementById('readLink');

class Carousel {
    constructor(container, item, itemg, control) {
        this.carouselContainer = container;
        this.carouselItem = [...item];
        this.carouselItemg = [...itemg];
        this.carouselControl = control;
        this.bookUrl = [
            // hang 1
            '/Tuổi Trẻ Đáng Giá Bao Nhiêu (Tái Bản 2021)',
            '/21 Bài Học Cho Thế Kỷ 21 (Tái Bản)',
            '/48 Nguyên Tắc Chủ Chốt Của Quyền Lực',
            '/Trẻ Thông Minh Nhờ Đúng Đắn Của Cha Mẹ',
            '/999 Lá Thư Gửi Cho Chính Mình (Tái Bản)',
            '/2666 – Roberto Bolaño (Tái Bản 2006)'
            // hang 2
        ,];
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

    setCurrentState(direction) {
        if (direction.classList.contains('container-controls-previous')) {
            this.carouselItem.unshift(this.carouselItem.pop());
            this.carouselItemg.unshift(this.carouselItemg.pop());
            this.bookUrl.unshift(this.bookUrl.pop());
        } else {
            this.carouselItem.push(this.carouselItem.shift());
            this.carouselItemg.push(this.carouselItemg.shift());
            this.bookUrl.push(this.bookUrl.shift());
        }
        this.updateItem();
        this.updateItemg();
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
                this.setCurrentState(control);
            });
        });
    }

    updateReadLink() {
        const currentLink = this.getCurrentLink();
        readLink.href = currentLink;
    }

    getCurrentLink() {
        return this.bookUrl[2] || '#'; 
    }

    autoslide() {
        setInterval(() => {
            this.carouselItem.push(this.carouselItem.shift());
            this.carouselItemg.push(this.carouselItemg.shift());
            this.bookUrl.push(this.bookUrl.shift());
            this.updateItem();
            this.updateItemg();
        },5000000);
    }
}

const runjs = new Carousel(containerItem, item, itemg, control);
runjs.updateItem();
runjs.updateItemg();
runjs.setControl();
runjs.useControl();
runjs.autoslide();

readLink.addEventListener('click', e => {
    e.preventDefault();
    const currentUrl = runjs.getCurrentLink();
    if (currentUrl !== '#') {
        window.location.href = currentUrl;
    }
});
