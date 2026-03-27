function openLogin() {
  document.getElementById("loginModal").classList.remove("hidden");
  document.getElementById("loginModal").classList.add("flex");
}

function closeLogin() {
  document.getElementById("loginModal").classList.add("hidden");
}

function analyzeCycle() {
  const cycle = document.getElementById("cycle").value;
  let msg = "";

  if (cycle < 21 || cycle > 35) {
    msg = "⚠️ Cycle appears irregular. Consider lifestyle or dietary adjustments.";
  } else {
    msg = "✅ Cycle length is within a healthy range.";
  }

  document.getElementById("cycleResult").innerText = msg;
}
document.getElementById("loginForm").addEventListener("submit", function (e) {
  e.preventDefault(); // prevent page reload

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  fetch("http://127.0.0.1:5000/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ email, password })
  })
  .then(res => {
    if (res.ok) {
      // ✅ redirect after successful login
      window.location.href = "/";

    } else {
      alert("Invalid email or password");
    }
  })
  .catch(err => {
    console.error(err);
    alert("Server error");
  });
});
