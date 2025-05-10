### LLM Controlled MCP


## 🧠 Goal

Automate the UI interaction with **Microsoft Paint** using a Language Model (LLM), where the **LLM issues structured function calls** (not code) to:

* Open Paint
* Draw a rectangle with specified coordinates
* Add text inside the rectangle
* Save the image
* Send it via email
* Exit the loop

---

## 🔧 Tools

| Tool Name                        | Description                                 |
| -------------------------------- | ------------------------------------------- |
| `open_paint()`                   | Launches Microsoft Paint                    |
| `draw_rectangle(x1, y1, x2, y2)` | Draws a rectangle on the canvas             |
| `add_text_in_paint(text)`        | Adds the specified text to the canvas       |
| `save_paint_image()`             | Saves the Paint canvas as an image          |
| `send_email_with_image(email)`   | Emails the saved image to the given address |

---

## ⚙️ System Prompt

```python
system_prompt1 = f"""Now, you are a UI agent of a laptop who manages and controls applications via function calls. You have access to the following tools:

Available tools:
{tools_description}

You must follow these instructions strictly:

1. All responses must be a single line starting with either:
   - FUNCTION_CALL: function_name|param1|param2|...
   - FINAL_ANSWER: [number]

2. Do not provide natural language explanations or comments.

3. Valid example calls:
   - FUNCTION_CALL: open_paint
   - FUNCTION_CALL: draw_rectangle|780|380|1140|700
   - FUNCTION_CALL: add_text_in_paint|1111
   - FUNCTION_CALL: save_paint_image
   - FUNCTION_CALL: send_email_with_image|someone@example.com

4. After calling open_paint, wait longer for Paint to be fully maximized:
   - # # Wait longer for Paint to be fully maximized
   - # await asyncio.sleep(1)

5. Once text is inserted into the rectangle and email is sent, **exit the loop** by BREAKING the session.

DO NOT repeat function calls with the same parameters.
DO NOT output anything other than structured FUNCTION_CALL lines.
"""
```

---

## 🗨️ User Prompt Template

```python
query = """Draw a rectangle with coordinates (780, 380) to (1140, 700) and then insert 1111 inside the rectangle of Paint application. After that, save the image and email it to me@example.com."""
```

---

## 📁 Project Structure

```
.
├── agent/
│   ├── paint_tools.py          # Tool implementations
│   └── ui_agent_runner.py      # LLM + tool executor
├── prompts/
│   ├── system_prompt.txt       # Instruction-only prompt
│   └── user_prompt_template.txt# User prompt with parameters
├── README.md                   # Full project description
```

---

## 📄 Full README.md

```markdown
# 🖼️ LLM-Controlled Paint Automation Agent

This project demonstrates how a Language Model (LLM) acts as a UI agent to control desktop applications like Microsoft Paint via structured prompts and function calls. The flow automates drawing a rectangle, inserting text, saving the image, and emailing the result — all through natural language instructions processed by the LLM.

## 🔄 Automated Flow

The full automation works as follows:

1. **Open Microsoft Paint**
2. **Wait for Paint to fully maximize**
3. **Draw a rectangle** with coordinates provided by the user (e.g., `(780, 380)` to `(1140, 700)`)
4. **Insert user-specified text** (e.g., `"1111"`) into the rectangle
5. **Save the image**
6. **Send the saved image as an email attachment**
7. **Exit the loop**

## 🧠 Powered by LLM + Function Calling

The LLM is prompted to control tools using structured function calls like:

         ```
         
         FUNCTION\_CALL: open\_paint

         FUNCTION\_CALL: draw\_rectangle|780|380|1140|700
         
         FUNCTION\_CALL: add\_text\_in\_paint|1111
         
         FUNCTION\_CALL: save\_paint\_image
         
         FUNCTION\_CALL: send\_email\_with\_image|[me@example.com](mailto:me@example.com)
         BREAK
         
         ```

## 🔧 Available Tools

| Tool Name                        | Description                                      |
|----------------------------------|--------------------------------------------------|
| `open_paint()`                   | Launches Microsoft Paint                         |
| `draw_rectangle(x1, y1, x2, y2)` | Draws a rectangle on the canvas                  |
| `add_text_in_paint(text)`        | Adds the specified text to the canvas            |
| `save_paint_image()`             | Saves the Paint canvas as an image               |
| `send_email_with_image(email)`   | Emails the saved image to the given address      |

## 🎯 Example User Prompt

```

Draw a rectangle with coordinates (780, 380) to (1140, 700) and then insert 1111 inside the rectangle of the Paint application. After that, save the image and email it to [me@example.com](mailto:me@example.com).

```

## ▶️ Demo Video

Watch the automation in action on YouTube:  
📺 [https://youtu.be/O5ER4n-ouHg](https://youtu.be/O5ER4n-ouHg)

---


## 📩 Setup Instructions

1. Ensure your environment supports GUI automation (e.g., Windows with `pyautogui`, `pygetwindow`, `pywin32`)
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
````



## ✅ Future Enhancements

* Add dynamic image file naming
* Include screenshot preview before sending emails
* Extend to support other desktop applications
* Add retry logic and error handling for tool failures


