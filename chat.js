// Simple localStorage-based chat history for StudyMate
function saveConversation(conversation) {
    localStorage.setItem('studymate_conversation', JSON.stringify(conversation));
}

function loadConversation() {
    const conv = localStorage.getItem('studymate_conversation');
    return conv ? JSON.parse(conv) : [];
}

function clearConversation() {
    localStorage.removeItem('studymate_conversation');
}

window.saveConversation = saveConversation;
window.loadConversation = loadConversation;
window.clearConversation = clearConversation;
