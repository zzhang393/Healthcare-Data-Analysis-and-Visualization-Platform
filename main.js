function submitQuery() {
    const query = document.getElementById("query-input").value;
    fetch('http://localhost:5000/query', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: query }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = JSON.stringify(data.results, null, 2);
    })
    .catch(error => console.error('Error:', error));
}
