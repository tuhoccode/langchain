    const galleryContainer = document.querySelector('.container-item'); 
    const galleryControlsContainer = document.querySelector('.container-controls');
    const textItem = document.querySelectorAll('.itemg')
    const galleryControls = ['previous', 'next']; 
    const galleryItems = document.querySelectorAll('.item');
    class Carousel {
        constructor(container, items, controls, itemg) { 
            this.carouselContainer = container; 
            this.carouselControls = controls;
            this.carouselArray = [...items];
            this.carouselItemg = [...itemg];
        }
    
        updateGallery() {
            this.carouselArray.forEach(el => {
                el.classList.remove('item-1', 'item-2', 'item-3', 'item-4', 'item-5');
            });
            this.carouselArray.slice(0, 5).forEach((el, i) => {
                el.classList.add(`item-${i + 1}`);
            });
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
            } else {
                this.carouselArray.push(this.carouselArray.shift());
                this.carouselItemg.push(this.carouselItemg.shift());
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
    }
    
    // Khởi tạo carousel
    const exampleCarousel = new Carousel(galleryContainer, galleryItems, galleryControls, textItem);
    exampleCarousel.updateGallery();
    exampleCarousel.updateItemg();
    exampleCarousel.setControls();
    exampleCarousel.useControls();
    








