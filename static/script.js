// ==========================================
// DIABETES PREDICTION SYSTEM - JAVASCRIPT
// Frontend Logic for Form Handling & API Calls
// ==========================================

document.addEventListener('DOMContentLoaded', function () {

    // ========== DOM ELEMENT REFERENCES ==========
    const predictionForm   = document.getElementById('predictionForm');
    const predictBtn       = document.getElementById('predictBtn');
    const resetBtn         = document.getElementById('resetBtn');
    const resultCard       = document.getElementById('resultCard');
    const errorCard        = document.getElementById('errorCard');
    const closeResult      = document.getElementById('closeResult');
    const closeError       = document.getElementById('closeError');

    // Result elements
    const resultIcon       = document.getElementById('resultIcon');
    const predictionText   = document.getElementById('predictionText');
    const confidenceText   = document.getElementById('confidenceText');
    const diabeticProb     = document.getElementById('diabeticProb');
    const nonDiabeticProb  = document.getElementById('nonDiabeticProb');
    const diabeticBar      = document.getElementById('diabeticBar');
    const nonDiabeticBar   = document.getElementById('nonDiabeticBar');
    const errorMessage     = document.getElementById('errorMessage');

    // Form inputs
    const genderInput        = document.getElementById('gender');
    const ageInput           = document.getElementById('age');
    const hypertensionInput  = document.getElementById('hypertension');
    const heartDiseaseInput  = document.getElementById('heart_disease');
    const smokingInput       = document.getElementById('smoking_history');
    const bmiInput           = document.getElementById('bmi');
    const hba1cInput         = document.getElementById('hba1c');
    const bloodGlucoseInput  = document.getElementById('blood_glucose');

    // ========== UTILITY FUNCTIONS ==========

    function showLoading() {
        predictBtn.disabled = true;
        predictBtn.style.opacity = '0.7';
        predictBtn.style.cursor = 'not-allowed';
    }

    function hideLoading() {
        predictBtn.disabled = false;
        predictBtn.style.opacity = '1';
        predictBtn.style.cursor = 'pointer';
    }

    function hideAllCards() {
        resultCard.style.display = 'none';
        errorCard.style.display  = 'none';
    }

    /**
     * Validate all form inputs before submission
     */
    function validateInputs(data) {
        const age         = parseFloat(data.age);
        const bmi         = parseFloat(data.bmi);
        const hba1c       = parseFloat(data.HbA1c_level);
        const bloodGlucose = parseFloat(data.blood_glucose_level);

        if (!data.gender) {
            return { valid: false, message: 'Please select a gender.' };
        }
        if (isNaN(age) || age < 0 || age > 120) {
            return { valid: false, message: 'Age must be between 0 and 120 years.' };
        }
        if (data.hypertension === '' || data.hypertension === undefined) {
            return { valid: false, message: 'Please select hypertension status.' };
        }
        if (data.heart_disease === '' || data.heart_disease === undefined) {
            return { valid: false, message: 'Please select heart disease status.' };
        }
        if (!data.smoking_history) {
            return { valid: false, message: 'Please select a smoking history.' };
        }
        if (isNaN(bmi) || bmi < 10 || bmi > 70) {
            return { valid: false, message: 'BMI must be between 10 and 70 kg/m².' };
        }
        if (isNaN(hba1c) || hba1c < 3 || hba1c > 15) {
            return { valid: false, message: 'HbA1c must be between 3.0% and 15.0%.' };
        }
        if (isNaN(bloodGlucose) || bloodGlucose < 0 || bloodGlucose > 500) {
            return { valid: false, message: 'Blood glucose must be between 0 and 500 mg/dL.' };
        }

        return { valid: true };
    }

    /**
     * Display prediction result
     */
    function displayResult(data) {
        hideAllCards();

        const isDiabetic = data.prediction === 'Diabetic';

        resultCard.classList.remove('diabetic');
        if (isDiabetic) resultCard.classList.add('diabetic');

        resultIcon.textContent      = isDiabetic ? '🔴' : '✅';
        predictionText.textContent  = data.prediction;
        predictionText.className    = isDiabetic ? 'diabetic' : 'non-diabetic';
        confidenceText.textContent  = `Confidence: ${data.confidence.toFixed(1)}%`;

        diabeticProb.textContent    = `${data.probability_diabetic.toFixed(1)}%`;
        nonDiabeticProb.textContent = `${data.probability_non_diabetic.toFixed(1)}%`;

        // Reset then animate bars
        diabeticBar.style.width    = '0%';
        nonDiabeticBar.style.width = '0%';
        setTimeout(() => {
            diabeticBar.style.width    = `${data.probability_diabetic}%`;
            nonDiabeticBar.style.width = `${data.probability_non_diabetic}%`;
        }, 100);

        resultCard.style.display = 'block';
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
        errorCard.style.display  = 'block';
        setTimeout(() => {
            errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 150);
    }

    // ========== EVENT HANDLERS ==========

    async function handleFormSubmit(event) {
        event.preventDefault();

        const formData = {
            gender:              genderInput.value,
            age:                 ageInput.value,
            hypertension:        hypertensionInput.value,
            heart_disease:       heartDiseaseInput.value,
            smoking_history:     smokingInput.value,
            bmi:                 bmiInput.value,
            HbA1c_level:         hba1cInput.value,
            blood_glucose_level: bloodGlucoseInput.value
        };

        console.log('Form data:', formData);

        const validation = validateInputs(formData);
        if (!validation.valid) {
            displayError(validation.message);
            return;
        }

        showLoading();
        hideAllCards();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            console.log('Response:', data);

            if (response.ok) {
                displayResult(data);
            } else {
                displayError(data.error || 'Prediction failed. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            displayError('Network error. Please check your connection and try again.');
        } finally {
            hideLoading();
        }
    }

    function handleReset() {
        predictionForm.reset();
        hideAllCards();
        diabeticBar.style.width    = '0%';
        nonDiabeticBar.style.width = '0%';
        document.querySelectorAll('.form-input').forEach(input => {
            input.style.borderColor = '';
        });
    }

    // ========== EVENT LISTENERS ==========
    if (predictionForm) predictionForm.addEventListener('submit', handleFormSubmit);
    if (resetBtn)       resetBtn.addEventListener('click', handleReset);
    if (closeResult)    closeResult.addEventListener('click', () => resultCard.style.display = 'none');
    if (closeError)     closeError.addEventListener('click',  () => errorCard.style.display  = 'none');

    // ========== REAL-TIME BORDER VALIDATION (numeric inputs only) ==========
    document.querySelectorAll('.form-input[type="number"]').forEach(input => {
        input.addEventListener('input', function () {
            const value = parseFloat(this.value);
            const min   = parseFloat(this.getAttribute('min'));
            const max   = parseFloat(this.getAttribute('max'));
            if (this.value && (value < min || value > max)) {
                this.style.borderColor = '#ef4444';
            } else {
                this.style.borderColor = '';
            }
        });
    });

    console.log('%c🏥 Diabetes Prediction System', 'color: #2563eb; font-size: 20px; font-weight: bold;');
    console.log('%cSystem Ready!', 'color: #10b981; font-size: 14px; font-weight: bold;');
});