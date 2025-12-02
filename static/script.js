document.getElementById("loanForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        gender: document.getElementById("gender").value,
        married: document.getElementById("married").value,
        dependents: document.getElementById("dependents").value,
        education: document.getElementById("education").value,
        self_employed: document.getElementById("self_employed").value,
        applicant_income: document.getElementById("applicant_income").value,
        coapplicant_income: document.getElementById("coapplicant_income").value,
        loan_amount: document.getElementById("loan_amount").value,
        loan_term: document.getElementById("loan_term").value,
        credit_history: document.getElementById("credit_history").value,
        property_area: document.getElementById("property_area").value
    };

    try {
        const res = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await res.json();

        document.getElementById("result").innerHTML =
            `<div class="alert alert-success">${result.prediction}</div>`;

    } catch (err) {
        document.getElementById("result").innerHTML =
            `<div class="alert alert-danger">Error: ${err}</div>`;
    }
});
