// ==========================================
// DIABETES PREDICTION SYSTEM - JAVASCRIPT
// Frontend Logic for Form Handling & API Calls
// ==========================================

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // ========== DOM ELEMENT REFERENCES ==========
    const predictionForm = document.getElementById('predictionForm');
    const predictBtn = document.getElementById('predictBtn');
    const resetBtn = document.getElementById('resetBtn');
    const resultCard = document.getElementById('resultCard');
    const errorCard = document.getElementById('errorCard');
    const closeResult = document.getElementById('closeResult');
    const closeError = document.getElementById('closeError');

    // Result elements
    const resultIcon = document.getElementById('resultIcon');
    const predictionText = document.getElementById('predictionText');
    const confidenceText = document.getElementById('confidenceText');
    const diabeticProb = document.getElementById('diabeticProb');
    const nonDiabeticProb = document.getElementById('nonDiabeticProb');
    const diabeticBar = document.getElementById('diabeticBar');
    const nonDiabeticBar = document.getElementById('nonDiabeticBar');
    const errorMessage = document.getElementById('errorMessage');

    // Form inputs
    const fpgInput = document.getElementById('fpg');
    const ogttInput = document.getElementById('ogtt');
    const randomPgInput = document.getElementById('random_pg');
    const hba1cInput = document.getElementById('hba1c');

    // ========== UTILITY FUNCTIONS ==========

    /**
     * Show loading state on predict button
     */
    function showLoading() {
        predictBtn.disabled = true;
        predictBtn.style.opacity = '0.7';
        predictBtn.style.cursor = 'not-allowed';
    }

    /**
     * Hide loading state on predict button
     */
    function hideLoading() {
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
        predictBtn.style.cursor = 'pointer';
    }

    /**
     * Hide all result and error cards
     */
    function hideAllCards() {
        resultCard.style.display = 'none';
        errorCard.style.display = 'none';
    }

    /**
     * Validate form inputs before submission
     */
    function validateInputs(data) {
        const fpg = parseFloat(data.FPG);
        const ogtt = parseFloat(data.OGTT);
        const randomPG = parseFloat(data.Random_Plasma_Glucose);
        const hba1c = parseFloat(data.HbA1c);

        if (isNaN(fpg) || fpg < 50 || fpg > 400) {
            return { valid: false, message: 'FPG must be between 50 and 400 mg/dL' };
        }

        if (isNaN(ogtt) || ogtt < 50 || ogtt > 500) {
            return { valid: false, message: 'OGTT must be between 50 and 500 mg/dL' };
        }

        if (isNaN(randomPG) || randomPG < 50 || randomPG > 500) {
            return { valid: false, message: 'Random Plasma Glucose must be between 50 and 500 mg/dL' };
        }

        if (isNaN(hba1c) || hba1c < 3 || hba1c > 15) {
            return { valid: false, message: 'HbA1c must be between 3.0% and 15.0%' };
        }

        return { valid: true };
    }

    /**
     * Display prediction result
     */
    function displayResult(data) {
        hideAllCards();

        const isDiabetic = data.prediction === 'Diabetic';

        // Update result card styling
        resultCard.classList.remove('diabetic');
        if (isDiabetic) {
            resultCard.classList.add('diabetic');
        }

        // Update icon
        resultIcon.textContent = isDiabetic ? '🔴' : '✅';

        // Update prediction text
        predictionText.textContent = data.prediction;
        predictionText.className = isDiabetic ? 'diabetic' : 'non-diabetic';

        // Update confidence
        confidenceText.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;

        // Update probability percentages
        diabeticProb.textContent = `${data.probability_diabetic.toFixed(1)}%`;
        nonDiabeticProb.textContent = `${data.probability_non_diabetic.toFixed(1)}%`;

        // Reset progress bars first
        diabeticBar.style.width = '0%';
        nonDiabeticBar.style.width = '0%';

        // Animate progress bars after a short delay
        setTimeout(() => {
            diabeticBar.style.width = `${data.probability_diabetic}%`;
            nonDiabeticBar.style.width = `${data.probability_non_diabetic}%`;
        }, 100);

        // Show result card
        resultCard.style.display = 'block';

        // Scroll to result smoothly
        setTimeout(() => {
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 150);
    }

    /**
     * Display error message
     */
    function displayError(message) {
        hideAllCards();
        errorMessage.textContent = message;
        errorCard.style.display = 'block';

        // Scroll to error
        setTimeout(() => {
            errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 150);
    }

    // ========== EVENT HANDLERS ==========

    /**
     * Handle form submission
     */
    async function handleFormSubmit(event) {
        event.preventDefault();
        
        console.log('Form submitted');

        // Collect form data
        const formData = {
            FPG: fpgInput.value,
            OGTT: ogttInput.value,
            Random_Plasma_Glucose: randomPgInput.value,
            HbA1c: hba1cInput.value
        };

        console.log('Form data:', formData);

        // Validate inputs
        const validation = validateInputs(formData);
        if (!validation.valid) {
            displayError(validation.message);
            return;
        }

        // Show loading state
        showLoading();
        hideAllCards();

        try {
            console.log('Sending prediction request...');

            // Send POST request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            console.log('Response status:', response.status);

            const data = await response.json();
            console.log('Response data:', data);

            if (response.ok) {
                // Display successful prediction
                displayResult(data);
            } else {
                // Display error from server
                displayError(data.error || 'Prediction failed. Please try again.');
            }

        } catch (error) {
            console.error('Error during prediction:', error);
            displayError('Network error. Please check your connection and try again.');
        } finally {
            // Hide loading state
            hideLoading();
        }
    }

    /**
     * Handle reset button click
     */
    function handleReset() {
        predictionForm.reset();
        hideAllCards();

        // Reset progress bars
        diabeticBar.style.width = '0%';
        nonDiabeticBar.style.width = '0%';

        // Reset input border colors
        const inputs = document.querySelectorAll('.form-input');
        inputs.forEach(input => {
            input.style.borderColor = '';
        });
    }

    /**
     * Close result card
     */
    function handleCloseResult() {
        resultCard.style.display = 'none';
    }

    /**
     * Close error card
     */
    function handleCloseError() {
        errorCard.style.display = 'none';
    }

    // ========== EVENT LISTENERS ==========

    if (predictionForm) {
        predictionForm.addEventListener('submit', handleFormSubmit);
    }

    if (resetBtn) {
        resetBtn.addEventListener('click', handleReset);
    }

    if (closeResult) {
        closeResult.addEventListener('click', handleCloseResult);
    }

    if (closeError) {
        closeError.addEventListener('click', handleCloseError);
    }

    // ========== INPUT VALIDATION FEEDBACK ==========

    const allInputs = document.querySelectorAll('.form-input');
    allInputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.getAttribute('min'));
            const max = parseFloat(this.getAttribute('max'));

            if (this.value && (value < min || value > max)) {
                this.style.borderColor = '#ef4444';
            } else {
                this.style.borderColor = '';
            }
        });
    });

    // ========== CONSOLE WELCOME MESSAGE ==========
    console.log('%c🏥 Diabetes Prediction System', 'color: #2563eb; font-size: 20px; font-weight: bold;');
    console.log('%cSystem Ready!', 'color: #10b981; font-size: 14px; font-weight: bold;');

});