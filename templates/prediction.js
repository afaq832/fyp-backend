document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent the default form submission

    const player = document.getElementById('player').value;
    const opposition = document.getElementById('opposition').value;
    const ballsFaced = document.getElementById('balls_faced').value;
    // const overs = document.getElementById('overs').value;

    // Construct the data to send in the request
    const data = {
        player: player,
        opposition: opposition,
        balls_faced: ballsFaced,
        overs: 20,
    };

    // Make a POST request to the Flask server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        // Display the prediction result using Bootstrap's alert for aesthetics
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerHTML = `<div class="alert alert-success" role="alert">${data.message}</div>`;
        predictionResult.style.display = 'block'; // Make sure the result is visible
    })
    .catch((error) => {
        console.error('Error:', error);
        // Optionally handle errors by displaying them in a similar styled alert
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerHTML = `<div class="alert alert-danger" role="alert">Error: Could not retrieve prediction.</div>`;
        predictionResult.style.display = 'block';
    });
});
