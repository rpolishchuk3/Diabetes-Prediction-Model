// ==========================================
// DIABETES RISK ASSESSMENT — JAVASCRIPT
// ==========================================

document.addEventListener('DOMContentLoaded', function () {

    const predictionForm    = document.getElementById('predictionForm');
    const predictBtn        = document.getElementById('predictBtn');
    const resetBtn          = document.getElementById('resetBtn');
    const resultCard        = document.getElementById('resultCard');
    const errorCard         = document.getElementById('errorCard');
    const closeResult       = document.getElementById('closeResult');
    const closeError        = document.getElementById('closeError');

    const predictionText    = document.getElementById('predictionText');
    const confidenceText    = document.getElementById('confidenceText');
    const diabeticProb      = document.getElementById('diabeticProb');
    const nonDiabeticProb   = document.getElementById('nonDiabeticProb');
    const diabeticBar       = document.getElementById('diabeticBar');
    const nonDiabeticBar    = document.getElementById('nonDiabeticBar');
    const errorMessage      = document.getElementById('errorMessage');

    const genderInput       = document.getElementById('gender');
    const ageInput          = document.getElementById('age');
    const hypertensionInput = document.getElementById('hypertension');
    const heartDiseaseInput = document.getElementById('heart_disease');
    const smokingInput      = document.getElementById('smoking_history');
    const bmiInput          = document.getElementById('bmi');
    const hba1cInput        = document.getElementById('hba1c');
    const bloodGlucoseInput = document.getElementById('blood_glucose');

    // ── Loading state ──
    function showLoading() {
        predictBtn.disabled = true;
        predictBtn.querySelector('.btn-inner').style.display = 'none';
        predictBtn.querySelector('.btn-loading').style.display = 'flex';
    }
    function hideLoading() {
        predictBtn.disabled = false;
        predictBtn.querySelector('.btn-inner').style.display = 'flex';
        predictBtn.querySelector('.btn-loading').style.display = 'none';
    }

    function hideAllCards() {
        resultCard.style.display = 'none';
        errorCard.style.display  = 'none';
    }

    // ── Validation ──
    function validateInputs(data) {
        const age          = parseFloat(data.age);
        const bmi          = parseFloat(data.bmi);
        const hba1c        = parseFloat(data.HbA1c_level);
        const bloodGlucose = parseFloat(data.blood_glucose_level);

        if (!data.gender)
            return { valid: false, message: 'Please select a gender.' };
        if (isNaN(age) || age < 0 || age > 120)
            return { valid: false, message: 'Age must be between 0 and 120 years.' };
        if (data.hypertension === '' || data.hypertension == null)
            return { valid: false, message: 'Please select hypertension status.' };
        if (data.heart_disease === '' || data.heart_disease == null)
            return { valid: false, message: 'Please select heart disease status.' };
        if (!data.smoking_history)
            return { valid: false, message: 'Please select a smoking history.' };
        if (isNaN(bmi) || bmi < 10 || bmi > 70)
            return { valid: false, message: 'BMI must be between 10 and 70 kg/m².' };
        if (isNaN(hba1c) || hba1c < 3 || hba1c > 15)
            return { valid: false, message: 'HbA1c must be between 3.0% and 15.0%.' };
        if (isNaN(bloodGlucose) || bloodGlucose < 0 || bloodGlucose > 500)
            return { valid: false, message: 'Blood glucose must be between 0 and 500 mg/dL.' };

        return { valid: true };
    }

    // ── Display result ──
    function displayResult(data) {
        hideAllCards();

        const isDiabetic = data.prediction === 'Diabetic';

        resultCard.className = 'result-card ' + (isDiabetic ? 'diabetic' : 'non-diabetic');
        predictionText.textContent = data.prediction;
        predictionText.className   = isDiabetic ? 'verdict-text diabetic' : 'verdict-text non-diabetic';
        confidenceText.textContent = `Confidence: ${Number(data.confidence).toFixed(1)}%`;

        diabeticProb.textContent    = `${Number(data.probability_diabetic).toFixed(1)}%`;
        nonDiabeticProb.textContent = `${Number(data.probability_non_diabetic).toFixed(1)}%`;

        diabeticBar.style.width    = '0%';
        nonDiabeticBar.style.width = '0%';
        setTimeout(() => {
            diabeticBar.style.width    = `${data.probability_diabetic}%`;
            nonDiabeticBar.style.width = `${data.probability_non_diabetic}%`;
        }, 80);

        resultCard.style.display = 'block';
        setTimeout(() => resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 120);
    }

    // ── Display error ──
    function displayError(message) {
        hideAllCards();
        errorMessage.textContent = message;
        errorCard.style.display  = 'flex';
        setTimeout(() => errorCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 120);
    }

    // ── Submit ──
    async function handleFormSubmit(e) {
        e.preventDefault();

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

        const validation = validateInputs(formData);
        if (!validation.valid) { displayError(validation.message); return; }

        showLoading();
        hideAllCards();

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const data = await response.json();
            if (response.ok) {
                displayResult(data);
            } else {
                displayError(data.error || 'Prediction failed. Please try again.');
            }
        } catch (err) {
            displayError('Network error. Please check your connection and try again.');
        } finally {
            hideLoading();
        }
    }

    // ── Reset ──
    function handleReset() {
        predictionForm.reset();
        hideAllCards();
        diabeticBar.style.width    = '0%';
        nonDiabeticBar.style.width = '0%';
        document.querySelectorAll('.field input').forEach(i => i.classList.remove('invalid'));
    }

    // ── Listeners ──
    if (predictionForm) predictionForm.addEventListener('submit', handleFormSubmit);
    if (resetBtn)       resetBtn.addEventListener('click', handleReset);
    if (closeResult)    closeResult.addEventListener('click', () => resultCard.style.display = 'none');
    if (closeError)     closeError.addEventListener('click',  () => errorCard.style.display  = 'none');

    // real-time number validation
    document.querySelectorAll('.field input[type="number"]').forEach(input => {
        input.addEventListener('input', function () {
            const v = parseFloat(this.value);
            const bad = this.value && (v < +this.min || v > +this.max);
            this.classList.toggle('invalid', bad);
        });
    });

    console.log('%c🏥 Diabetes Risk Assessment', 'color:#1d4ed8;font-size:18px;font-weight:600');
    console.log('%cReady', 'color:#16a34a;font-weight:600');
});