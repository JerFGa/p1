document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('toggleSidebar');
    const toggleIcon = document.getElementById('toggleIcon');

    if (toggleBtn) {
        toggleBtn.addEventListener('click', function() {
            sidebar.classList.toggle('sidebar-collapsed');
            if (sidebar.classList.contains('sidebar-collapsed')) {
                toggleIcon.style.transform = 'rotate(180deg)';
            } else {
                toggleIcon.style.transform = 'rotate(0deg)';
            }
        });
    }
});
