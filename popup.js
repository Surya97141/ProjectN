document.getElementById('checkButton').addEventListener('click', function() {
  let newsClaim = document.getElementById('newsInput').value;
  
  if (newsClaim) {
    // Example API call to Gradio or your backend
    fetch("YOUR_GRADIO_API_URL", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ statement: newsClaim })
    })
    .then(response => response.json())
    .then(data => {
      document.getElementById('result').textContent = data.fake_news;
    })
    .catch(error => {
      console.error('Error:', error);
      document.getElementById('result').textContent = "Error detecting news.";
    });
  } else {
    document.getElementById('result').textContent = "Please enter a news claim.";
  }
});
