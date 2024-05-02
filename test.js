fetch("http://127.0.0.1:5000/predict_cpi", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ exchange_rate: 70000.0 }),
})
  .then((response) => response.json())
  .then((data) => console.log(data.predicted_cpi));

fetch("http://127.0.0.1:5000/predict_exchange_rate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ date: "2025-12-31" }),
})
  .then((response) => response.json())
  .then((data) => console.log(data.predicted_exchange_rate));
