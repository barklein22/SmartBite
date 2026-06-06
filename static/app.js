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

  // Markdown-style links from the backend, e.g. [פתח ב-Google Maps](https://...)
  // are converted to clickable links while keeping all other text escaped.
  safeText = safeText.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  // Raw URLs are also made clickable, as a fallback.
  safeText = safeText.replace(/(^|\s)(https?:\/\/[^\s<]+)/g,
    '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
  );

  safeText = safeText.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  safeText = safeText.replace(/\n/g, "<br>");
  return safeText;
}

function scrollToBottom() {
  chatBox.scrollTop = chatBox.scrollHeight;
}

function addMessage(text, sender) {
  const message = document.createElement("div");
  message.className = `message ${sender}`;
  message.innerHTML = formatMessageText(text);
  chatBox.appendChild(message);
  scrollToBottom();
  return message;
}

function showTypingIndicator() {
  const bubble = document.createElement("div");
  bubble.className = "message bot typing-bubble";
  bubble.setAttribute("aria-label", "SmartBite is typing");
  bubble.innerHTML = `
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
    <span class="typing-dot"></span>
  `;
  chatBox.appendChild(bubble);
  scrollToBottom();
  return bubble;
}

function removeTypingIndicator(bubble) {
  if (bubble && bubble.parentNode) {
    bubble.parentNode.removeChild(bubble);
  }
}

function formatBotResponse(result) {
  if (!result || !result.message) {
    return "אני לא בטוחה שהבנתי לגמרי 😊 אפשר לנסח שוב מה את מחפשת בעולם המסעדות?";
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

  const typingBubble = showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    });

    const result = await response.json();
    removeTypingIndicator(typingBubble);
    addMessage(formatBotResponse(result), "bot");
  } catch (error) {
    console.error(error);
    removeTypingIndicator(typingBubble);
    addMessage("משהו השתבש בשליחת ההודעה. תבדקי שהשרת עדיין רץ ושמפתחות ה־API מוגדרים ב־Render.", "bot");
  } finally {
    sendBtn.disabled = false;
    sendBtn.innerText = "Send";
    userInput.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", function(event) {
  if (event.key === "Enter") sendMessage();
});
