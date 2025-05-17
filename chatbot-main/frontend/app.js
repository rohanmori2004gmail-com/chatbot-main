class Chatbox {

    constructor() {
        this.args = {
            openButton: document.querySelector('#ask-to-bot'),
            openButtons: document.querySelector('.chatbox_button'),
            chatBox: document.querySelector('.gec_chatbot'),
            sendButton: document.querySelector('.send_button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const { openButton, openButtons,chatBox, sendButton } = this.args;

        openButtons.addEventListener('click', (event) => {
            event.preventDefault();
            this.toggleState(chatBox)
        });

        openButton.addEventListener('click', (event) => {
            event.preventDefault();
            this.toggleState(chatBox)
        });

        sendButton.addEventListener('click', () => this.onSendButton(chatBox));

        const node = chatBox.querySelector('input');
        node.addEventListener("keyup", ({ key }) => {
            if (key === "Enter") {
                this.onSendButton(chatBox)
            }
        })
    }

    toggleState(chatbox) {
        this.state = !this.state;

        if (this.state) {
            chatbox.classList.add('chatbox-active');
        } else {
            chatbox.classList.remove('chatbox-active');
        }
    }

    onSendButton(chatbox) {
        var textField = chatbox.querySelector('input');
        let text1 = textField.value
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 }
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ message: text1 }),
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
            .then(r => r.json())
            .then(r => {
                let msg2 = { name: "GEC_BOT", message: r.answer };
                this.messages.push(msg2);
                this.updateChatText(chatbox)
                textField.value = ''
            }).catch((error) => {
                console.error('Error:', error);
                this.updateChatText(chatbox)
                textField.value = ''
            });
    }

    updateChatText(chatbox) {
        var html = '';
        this.messages.slice().reverse().forEach(function (item, index) {
            if (item.name === "GEC_BOT") {
                html += '<div class="messages_item messages_item-visitor">' + item.message + '</div>'
            } else {
                html += '<div class="messages_item messages_item-operator">' + item.message + '</div>'
            }
        });

        const chatmessage = chatbox.querySelector('.chatbox_messages');
        chatmessage.innerHTML = html;
    }
}

const chatbox = new Chatbox();
chatbox.display();

document.addEventListener('DOMContentLoaded', () => {
    const faqQuestions = document.querySelectorAll('.faq-question');
  
    faqQuestions.forEach(question => {
      question.addEventListener('click', () => {
        const answer = question.nextElementSibling;
        if (answer.style.display === 'block') {
          answer.style.display = 'none';
        } else {
          answer.style.display = 'block';
        }
      });
    });
  });
