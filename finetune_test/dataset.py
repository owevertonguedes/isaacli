TOUCH_CODE = '''```csharp
using UnityEngine;

public class PlayerController2D : MonoBehaviour
{
    public float moveSpeed = 5f;
    public float jumpForce = 10f;

    private Rigidbody2D rb;
    private bool isTouchingScreen;
    private float touchDirectionX;

    void Start()
    {
        rb = GetComponent<Rigidbody2D>();
    }

    void Update()
    {
        // Le toque na tela (nao teclado) para mover e pular no mobile
        isTouchingScreen = Input.touchCount > 0;

        if (isTouchingScreen)
        {
            Touch touch = Input.GetTouch(0);
            // Metade esquerda da tela move, metade direita pula
            touchDirectionX = touch.position.x < Screen.width / 2f ? -1f : 1f;

            if (touch.position.x < Screen.width / 2f)
            {
                rb.velocity = new Vector2(touchDirectionX * moveSpeed, rb.velocity.y);
            }
            else if (touch.phase == TouchPhase.Began)
            {
                rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
            }
        }
        else
        {
            rb.velocity = new Vector2(0f, rb.velocity.y);
        }
    }
}
```'''

REQUESTS = [
    "Escreva um script C# para Unity de um controle de personagem 2D simples para um jogo mobile (iOS/Android): movimento lateral com toque na tela, e um pulo com física básica usando Rigidbody2D. Comente as partes principais.",
    "Preciso de um script Unity C# pra mover um personagem 2D num jogo de celular usando o dedo na tela, com pulo via Rigidbody2D.",
    "Cria um controlador de player mobile em Unity, movimento e pulo controlados por toque, não por teclado.",
    "Script Unity C# para plataforma mobile: personagem anda e pula ao tocar na tela (touch), física com Rigidbody2D.",
    "Faz um PlayerController pra Unity que funcione em Android/iOS, usando toque na tela pra mover e pular, sem depender de teclado ou joystick físico.",
    "Quero um código Unity de movimento de personagem 2D mobile: toque do lado esquerdo move, toque do lado direito pula.",
    "Gera um script de controle de jogador em Unity C# voltado pra celular (touch input), com pulo usando Rigidbody2D.",
    "Como faço um personagem 2D andar e pular em um jogo mobile Unity usando toque na tela?",
]

def build_examples():
    examples = []
    for req in REQUESTS:
        examples.append({
            "messages": [
                {"role": "user", "content": req},
                {"role": "assistant", "content": TOUCH_CODE},
            ]
        })
    return examples
