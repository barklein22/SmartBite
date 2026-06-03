const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatMessageText(text) {
  let safeText = escapeHtml(text);
  safeText = safeText.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  safeText = safeText.replace(/\n/g, "<br>");
  return safeText;
}

function addMessage(text, sender) {
  const message = document.createElement("div");
  message.className = `message ${sender}`;
  message.innerHTML = formatMessageText(text);
  chatBox.appendChild(message);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function formatBotResponse(result) {
  if (!result || !result.message) {
    return "אני לא בטוח שהבנתי לגמרי 😊 אפשר לנסח לי שוב מה את מחפשת?";
  }
  return result.message;
}

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message) return;

  addMessage(message, "user");
  userInput.value = "";
  sendBtn.disabled = true;
  sendBtn.innerText = "...";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const result = await response.json();
    addMessage(formatBotResponse(result), "bot");
  } catch (error) {
    console.error(error);
    addMessage("משהו השתבש בשליחת ההודעה. תבדקי שהשרת עדיין רץ ב־VS Code.", "bot");
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerText = "Send";
    userInput.focus();
  }
}

userInput.addEventListener("keydown", function(event) {
  if (event.key === "Enter") sendMessage();
});
