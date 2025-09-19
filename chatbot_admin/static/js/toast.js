// Toast Message System
class ToastManager {
    constructor() {
        this.container = this.createContainer();
    }

    createContainer() {
        let container = document.querySelector('.toast-container');
        if (!container) {
            container = document.createElement('div');
            container.className = 'toast-container';
            document.body.appendChild(container);
        }
        return container;
    }

    show(message, type = 'info', duration = 3000) {
        const toast = this.createToast(message, type);
        this.container.appendChild(toast);

        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 100);

        // Auto remove
        setTimeout(() => this.remove(toast), duration);

        return toast;
    }

    createToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-header">
                <strong>
                    ${this.getIcon(type)} ${this.getTitle(type)}
                </strong>
                <button type="button" class="btn-close" onclick="window.toastManager.remove(this.closest('.toast'))">&times;</button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        `;
        return toast;
    }

    getIcon(type) {
        const icons = {
            success: '✓',
            error: '✗',
            warning: '⚠',
            info: 'ℹ'
        };
        return icons[type] || icons.info;
    }

    getTitle(type) {
        const titles = {
            success: 'Thành công',
            error: 'Lỗi',
            warning: 'Cảnh báo',
            info: 'Thông báo'
        };
        return titles[type] || titles.info;
    }

    remove(toast) {
        toast.classList.add('hide');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }

    success(message, duration = 3000) {
        return this.show(message, 'success', duration);
    }

    error(message, duration = 3000) {
        return this.show(message, 'error', duration);
    }

    warning(message, duration = 3000) {
        return this.show(message, 'warning', duration);
    }

    info(message, duration = 3000) {
        return this.show(message, 'info', duration);
    }
}

// Initialize global toast manager
window.toastManager = new ToastManager();

// Django Messages Integration
document.addEventListener('DOMContentLoaded', function() {
    console.log('Toast system loaded'); // Debug log

    // Show Django messages as toasts
    const messages = document.querySelectorAll('.django-messages .alert');
    console.log('Found messages:', messages.length); // Debug log

    messages.forEach(function(message) {
        let type = 'info';
        if (message.classList.contains('alert-success')) type = 'success';
        else if (message.classList.contains('alert-danger')) type = 'error';
        else if (message.classList.contains('alert-warning')) type = 'warning';
        else if (message.classList.contains('alert-error')) type = 'error';

        window.toastManager.show(message.textContent.trim(), type);
        message.style.display = 'none';
    });
});