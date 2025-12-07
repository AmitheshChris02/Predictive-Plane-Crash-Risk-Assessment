// ===================================================================
// PREMIUM CINEMATIC JAVASCRIPT - LUXURY INTERACTIONS
// Liquid transitions, organic motion, parallax, bokeh particles
// ===================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all premium features
    initBokehParticles();
    initParallaxScrolling();
    initSmoothAnimations();
    initMicroInteractions();
    initShimmerEffects();
    initLiquidTransitions();
    
    // Initialize Bootstrap components
    initBootstrapComponents();
    
    // Form validation
    initFormValidation();
    
    // Auto-hide flash messages
    initFlashMessages();
    
    // Smooth scroll for anchor links
    initSmoothScroll();
});

// ===== BOKEH PARTICLES - DYNAMIC AMBIENT BACKGROUND =====
function initBokehParticles() {
    const container = document.createElement('div');
    container.id = 'bokeh-container';
    document.body.appendChild(container);
    
    const particleCount = 15;
    const colors = [
        'rgba(255, 107, 53, 0.15)',   // sunset orange
        'rgba(212, 175, 55, 0.2)',    // royal gold
        'rgba(15, 76, 117, 0.15)',    // deep teal
        'rgba(45, 134, 89, 0.15)',    // emerald
        'rgba(255, 255, 255, 0.1)'    // white
    ];
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'bokeh-particle';
        
        const size = Math.random() * 150 + 50; // 50-200px
        const color = colors[Math.floor(Math.random() * colors.length)];
        const startX = Math.random() * 100;
        const startY = Math.random() * 100;
        const duration = Math.random() * 10 + 15; // 15-25s
        const delay = Math.random() * 5;
        
        particle.style.width = size + 'px';
        particle.style.height = size + 'px';
        particle.style.background = `radial-gradient(circle, ${color} 0%, transparent 70%)`;
        particle.style.left = startX + '%';
        particle.style.top = startY + '%';
        particle.style.animationDuration = duration + 's';
        particle.style.animationDelay = delay + 's';
        
        // Add random float variation
        const floatVariation = Math.random() * 100 - 50;
        particle.style.setProperty('--float-x', floatVariation + 'px');
        particle.style.setProperty('--float-y', floatVariation + 'px');
        
        container.appendChild(particle);
    }
}

// ===== PARALLAX SCROLLING - DEPTH-BASED =====
function initParallaxScrolling() {
    // Only apply parallax to hero section, not cards (to prevent alignment issues)
    const parallaxElements = document.querySelectorAll('.hero-inner');
    
    let ticking = false;
    
    function updateParallax() {
        const scrollY = window.pageYOffset;
        
        parallaxElements.forEach((element) => {
            const rect = element.getBoundingClientRect();
            const elementTop = rect.top + scrollY;
            const windowHeight = window.innerHeight;
            
            // Only apply parallax when element is in viewport
            if (rect.top < windowHeight && rect.bottom > 0) {
                const speed = 0.05; // Subtle parallax
                const yPos = -(scrollY - elementTop + windowHeight) * speed;
                element.style.transform = `translateY(${yPos}px)`;
            } else {
                element.style.transform = '';
            }
        });
        
        ticking = false;
    }
    
    window.addEventListener('scroll', function() {
        if (!ticking) {
            window.requestAnimationFrame(updateParallax);
            ticking = true;
        }
    });
    
    // Initial call
    updateParallax();
}

// ===== SMOOTH ANIMATIONS - BUTTERY TRANSITIONS =====
function initSmoothAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe cards and sections
    const animatedElements = document.querySelectorAll('.card, .feature-card, .about-section, section');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
    
    // Stagger animation for feature cards
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.animationDelay = (index * 0.1) + 's';
        card.classList.add('zoom-in');
    });
}

// ===== MICRO-INTERACTIONS - LUXURIOUS FEEDBACK =====
function initMicroInteractions() {
    // Button ripple effect
    const buttons = document.querySelectorAll('.btn');
    buttons.forEach(button => {
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            button.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Card hover elevation
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        });
    });
    
    // Nav link active state
    const navLinks = document.querySelectorAll('.nav-link');
    const currentPath = window.location.pathname;
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
}

// ===== SHIMMER EFFECTS - PREMIUM GLOW =====
function initShimmerEffects() {
    // Add shimmer to titles
    const titles = document.querySelectorAll('h1, h2, h3, .card-title');
    titles.forEach(title => {
        title.addEventListener('mouseenter', function() {
            this.style.textShadow = '0 0 30px rgba(212, 175, 55, 0.6)';
            this.style.transition = 'text-shadow 0.3s ease';
        });
        
        title.addEventListener('mouseleave', function() {
            this.style.textShadow = '';
        });
    });
    
    // Shimmer animation for hero title
    const heroTitle = document.querySelector('.hero-inner h1');
    if (heroTitle) {
        setInterval(() => {
            heroTitle.style.animation = 'none';
            setTimeout(() => {
                heroTitle.style.animation = 'shimmer 3s ease-in-out infinite';
            }, 10);
        }, 3000);
    }
}

// ===== LIQUID TRANSITIONS - ORGANIC MOTION =====
function initLiquidTransitions() {
    // Smooth page transitions
    const links = document.querySelectorAll('a[href^="/"], a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.getAttribute('href').startsWith('#')) {
                return; // Let smooth scroll handle anchor links
            }
            
            // Add fade out effect
            document.body.style.transition = 'opacity 0.3s ease';
            document.body.style.opacity = '0.8';
        });
    });
    
    // Restore opacity on page load
    window.addEventListener('load', function() {
        document.body.style.opacity = '1';
    });
}

// ===== BOOTSTRAP COMPONENTS INITIALIZATION =====
function initBootstrapComponents() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
}

// ===== FORM VALIDATION =====
function initFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// ===== FLASH MESSAGES =====
function initFlashMessages() {
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
}

// ===== SMOOTH SCROLL =====
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#' || href === '') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ===== UTILITY FUNCTIONS =====

// Format date/time
function formatDateTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

// Show loading spinner
function showLoadingSpinner(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = '<div class="d-flex justify-content-center"><div class="spinner-border text-gold" role="status"><span class="visually-hidden">Loading...</span></div></div>';
    }
}

// Hide loading spinner
function hideLoadingSpinner(elementId, content) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = content || '';
    }
}

// Update video feed
function updateVideoFeed() {
    const videoFeed = document.getElementById('video-feed');
    if (videoFeed) {
        videoFeed.src = '/video_feed?' + new Date().getTime();
    }
}

// Update detection statistics
function updateDetectionStats() {
    fetch('/detection_results')
        .then(response => response.json())
        .then(data => {
            const birdCount = document.getElementById('bird-count');
            const potholeCount = document.getElementById('pothole-count');
            const statusElement = document.getElementById('processing-status');
            
            if (birdCount) {
                birdCount.textContent = data.bird_count;
                birdCount.classList.add('zoom-in');
                setTimeout(() => birdCount.classList.remove('zoom-in'), 600);
            }
            if (potholeCount) {
                potholeCount.textContent = data.pothole_count;
                potholeCount.classList.add('zoom-in');
                setTimeout(() => potholeCount.classList.remove('zoom-in'), 600);
            }
            
            if (statusElement) {
                if (data.processing) {
                    statusElement.innerHTML = '<span class="badge bg-warning">Processing...</span>';
                    updateVideoFeed();
                } else if (data.bird_count > 0 || data.pothole_count > 0) {
                    statusElement.innerHTML = '<span class="badge bg-success">Processing Complete</span>';
                } else {
                    statusElement.innerHTML = '<span class="badge bg-secondary">No video loaded</span>';
                }
            }
        })
        .catch(error => console.error('Error updating stats:', error));
}

// ===== ADD RIPPLE EFFECT STYLES =====
const style = document.createElement('style');
style.textContent = `
    .btn {
        position: relative;
        overflow: hidden;
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: scale(0);
        animation: ripple-animation 0.6s ease-out;
        pointer-events: none;
    }
    
    @keyframes ripple-animation {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ===== PERFORMANCE OPTIMIZATION =====
// Throttle scroll events
function throttle(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Apply throttling to scroll-heavy functions
const throttledParallax = throttle(() => {
    // Parallax updates are already optimized with requestAnimationFrame
}, 16);
