function toggleForms(formId) {
    const forms = ['loginForm', 'registerForm', 'forgotPasswordForm', 'otpForm'];
    forms.forEach(id => {
        document.getElementById(id).classList.add('hidden');
    });
    document.getElementById(formId).classList.remove('hidden');
}