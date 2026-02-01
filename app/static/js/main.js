// Small UX enhancement: highlight active input
const inputs = document.querySelectorAll("input");

inputs.forEach(input => {
    input.addEventListener("focus", () => {
        input.style.backgroundColor = "#f0f8ff";
    });

    input.addEventListener("blur", () => {
        input.style.backgroundColor = "white";
    });
});
