
const form = document.getElementById('news-form');

let currentUrl = "";

document.addEventListener('DOMContentLoaded', function() {
  chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
      currentUrl = tabs[0].url;
      document.getElementById('url-display').textContent = currentUrl;
  });
});

form.addEventListener('submit', async (event) => {

  event.preventDefault();


  //const input = document.getElementById('news-text').value;


  console.log("Sending input:", currentUrl);
  

  try {
 
    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({text: currentUrl}),
    });


    if (response.ok) {
 
      const data = await response.json();
      const prediction = data.prediction;
      const prediction_proba = data.prediction_proba;
      const resultDiv = document.getElementById('prediction-result');
      const predictionDiv = document.getElementById('prediction-proba');
      
      resultDiv.innerText = prediction === 0 ? 'The website seems to be Real' : 'The website seems to be Spam !!!';
      const probabilityPercentage = (prediction_proba[1] * 100).toFixed(0);
      predictionDiv.innerText = 'Risk Score : ' + probabilityPercentage + '%';
    } else {
      console.error('Request failed:', response.status);
    }
  } catch (error) {
    console.error('Request failed:', error);
  }
});
