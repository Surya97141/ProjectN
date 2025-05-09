document.addEventListener('DOMContentLoaded', function() {
  // Add event listener to the button after DOM is loaded
  document.getElementById('checkButton').addEventListener('click', function() {
    let newsClaim = document.getElementById('newsInput').value;

    if (newsClaim) {
      // API call to your Gradio backend hosted on Hugging Face
      fetch("https://su07rya-fakenews01.hf.space/gradio_api/run/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          "data": [newsClaim]  // Gradio expects data as a list
        })
      })
      .then(async response => {
        if (!response.ok) {
          // Enhanced error message with status and statusText
          throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`);
        }

        const data = await response.json();
        console.log("API Response:", data);

        // Checking if data is in the expected format
        if (data && data.data && Array.isArray(data.data) && data.data.length > 0) {
          // Display the prediction result
          document.getElementById('result').textContent = `Prediction: ${data.data[0]}`;
        } else {
          console.error("Unexpected response format:", data);
          document.getElementById('result').textContent = "Unexpected response format.";
        }
      })
      .catch(error => {
        console.error('Error:', error.message);
        document.getElementById('result').textContent = `Error detecting news: ${error.message}`;
      });
    } else {
      document.getElementById('result').textContent = "Please enter a news claim.";
    }
  });
});
