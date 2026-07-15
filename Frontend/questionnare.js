// questionnaire.js

const nextButtons = document.querySelectorAll('.next');
const prevButtons = document.querySelectorAll('.prev');
const submitButton = document.querySelector('.submit');
const cards = document.querySelectorAll('.card');

let currentCard = 0;

// Show next card
nextButtons.forEach(button => {
  button.addEventListener('click', () => {
    cards[currentCard].classList.remove('active');
    currentCard++;
    cards[currentCard].classList.add('active');
  });
});

// Show previous card
prevButtons.forEach(button => {
  button.addEventListener('click', () => {
    cards[currentCard].classList.remove('active');
    currentCard--;
    cards[currentCard].classList.add('active');
  });
});

// Submit action
submitButton.addEventListener('click', () => {
  alert('Thank you for completing the questionnaire! 🎉');
  // Redirect to dashboard or homepage
  window.location.href = "index.html"; // Change this to your dashboard page
});
