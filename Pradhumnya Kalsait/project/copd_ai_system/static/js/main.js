let storedFile = null;

// ===================== STAGE 1 =====================
document.getElementById("stage1Form").addEventListener("submit", async function (e) {
    e.preventDefault();

    const fileInput = document.getElementById("breathFile");

    if (!fileInput.files.length) {
        alert("Upload breath file");
        return;
    }

    storedFile = fileInput.files[0];

    const formData = new FormData();
    formData.append("file", storedFile);

    const response = await fetch("/predict-stage1", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    displayStage1Result(data);
});


function displayStage1Result(data) {

    const container = document.getElementById("stage1Result");

    container.innerHTML = `
        <p><strong>Prediction:</strong> ${data.prediction}</p>
        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
    `;

    // Only show Stage-2 if COPD
    if (data.prediction === "COPD") {
        document.getElementById("stage2Section").style.display = "block";
    } else {
        document.getElementById("stage2Section").style.display = "none";
    }
}


// ===================== STAGE 2 =====================
document.getElementById("stage2Form").addEventListener("submit", async function (e) {
    e.preventDefault();

    const inputs = e.target.querySelectorAll("input[type=number]");

    const clinicalData = {};
    inputs.forEach(input => {
        clinicalData[input.name] = parseFloat(input.value);
    });

    const response = await fetch("/predict-stage2", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(clinicalData)
    });

    const data = await response.json();

    displayStage2Result(data);
});


function displayStage2Result(data) {

    const container = document.getElementById("stage2Result");

    container.innerHTML = `
        <p><strong>GOLD Stage:</strong> ${data.gold_stage}</p>
        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
    `;
}