import subprocess
import os

def render_latex_to_pdf(latex_expression, filename="output"):
    """
    Renders a LaTeX expression into a PDF file.

    Args:
        latex_expression (str): The LaTeX code to render.
        filename (str): The base name for the output PDF file (e.g., "my_equation").
                        The actual file will be filename.pdf.
    """
    # Create a minimal LaTeX document
    latex_document = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amsfonts}}
\\usepackage{{amssymb}}
\\usepackage{{graphicx}} % For graphics if needed, good to include
\\pagestyle{{empty}} % No page numbers

\\begin{{document}}
{latex_expression}
\\end{{document}}
"""

    tex_filename = f"{filename}.tex"
    pdf_filename = f"{filename}.pdf"

    try:
        # Write the LaTeX document to a .tex file
        with open(tex_filename, "w") as f:
            f.write(latex_document)

        print(f"LaTeX file '{tex_filename}' created.")

        # Compile the LaTeX file to PDF using pdflatex
        # We redirect stdout and stderr to avoid cluttering the console
        # You might want to remove this for debugging
        process = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_filename],
            capture_output=True,
            text=True
        )

        if process.returncode == 0:
            print(f"Successfully compiled '{tex_filename}' to '{pdf_filename}'.")
            # You can optionally open the PDF here
            # For Windows: os.startfile(pdf_filename)
            # For macOS: subprocess.run(["open", pdf_filename])
            # For Linux: subprocess.run(["xdg-open", pdf_filename])
        else:
            print(f"Error compiling LaTeX. Return code: {process.returncode}")
            print("--- pdflatex stdout ---")
            print(process.stdout)
            print("--- pdflatex stderr ---")
            print(process.stderr)
            print(f"Check '{filename}.log' for more details.")

    except FileNotFoundError:
        print("Error: 'pdflatex' command not found.")
        print("Please ensure you have a LaTeX distribution (like TeX Live or MiKTeX) installed and 'pdflatex' is in your system's PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up auxiliary files generated by pdflatex
        for ext in [".aux", ".log", ".out", ".fls", ".fdb_latexmk"]:
            if os.path.exists(f"{filename}{ext}"):
                os.remove(f"{filename}{ext}")
        print(f"Cleaned up auxiliary files for '{filename}'.")


if __name__ == "__main__":
    # Example 1: A simple equation
    equation = "$$E=mc^2$$"
    render_latex_to_pdf(equation, filename="physics_equation")

    # Example 2: A more complex mathematical expression
    complex_math = r"""
    \begin{align*}
    \int_0^\infty e^{-x^2} dx &= \frac{\sqrt{\pi}}{2} \\
    \sum_{n=1}^\infty \frac{1}{n^2} &= \frac{\pi^2}{6}
    \end{align*}
    """
    render_latex_to_pdf(complex_math, filename="complex_math")

    # Example 3: A short paragraph with some inline math
    text_with_math = r"""
    \section*{Introduction to LaTeX Expressions}
    This is an example of a paragraph containing inline mathematical expressions like $a^2 + b^2 = c^2$.
    We can also display more complex equations in a new line:
    \[
    \frac{d}{dx} \left( \int_{a}^{x} f(t) dt \right) = f(x)
    \]
    LaTeX is very powerful for typesetting documents.
    """
    render_latex_to_pdf(text_with_math, filename="paragraph_with_math")