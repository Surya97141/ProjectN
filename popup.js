document.getElementById('checkButton').addEventListener('click', function() {
  let newsClaim = document.getElementById('newsInput').value;
  
  if (newsClaim) {
    // API call to your Gradio backend hosted on Hugging Face
    fetch("https://su07rya-fakenews01.hf.space/run/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ 
        "data": [newsClaim]    // Gradio expects data as a list
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data && data.data && data.data.length > 0) {
        document.getElementById('result').textContent = data.data[0];
      } else {
        document.getElementById('result').textContent = "Unexpected response format.";
      }
    })
    .catch(error => {
      console.error('Error:', error);
      document.getElementById('result').textContent = "Error detecting news.";
    });
  } else {
    document.getElementById('result').textContent = "Please enter a news claim.";
  }
});
