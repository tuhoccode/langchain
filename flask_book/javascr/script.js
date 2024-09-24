function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const content = document.querySelector('.content');
    
    if (sidebar.style.transform === 'translateX(0%)') {
        sidebar.style.transform = 'translateX(-100%)';
        content.style.marginLeft = '0';
    } else {
        sidebar.style.transform = 'translateX(0%)';
        content.style.marginLeft = '250px';
    }
}
        
