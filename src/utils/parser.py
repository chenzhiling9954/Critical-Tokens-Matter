"""
The code implementation mainly refers to https://github.com/microsoft/ToRA/tree/main
"""
import json
import re


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def strip_string(string):
    string = str(string).strip()
    string = string.replace("\n", "")

    string = string.rstrip(".")

    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    string = string.replace("\\cdot", "")

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace("\"", "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    string = _fix_fracs(string)

    string = _fix_a_slash_b(string)

    return string

def extract_latex_answer(answer_string):
    pred = None
    if "Final Answer:" in answer_string:
        if "I hope it is correct" in answer_string:
            answer_string = answer_string.split("I hope it is correct")[0]
        final_answer_str = answer_string.split("Final Answer:")[-1]
        pred_list = re.findall("\$(.*)\$", final_answer_str, re.DOTALL)
        if len(pred_list) > 0:
            pred = pred_list[0]
    if pred is None:
        if 'boxed' in answer_string:
            ans = answer_string.split('boxed')[-1]
            if len(ans) == 0:
                return ""
            elif (ans[0] == '{'):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if (c == '{'):
                        stack += 1
                        a += c
                    elif (c == '}'):
                        stack -= 1
                        if (stack == 0): break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split('$')[0].strip()
            pred = a
        elif 'he answer is' in answer_string:
            pred = answer_string.split('he answer is')[-1].strip()
        else:
            pred_str = answer_string.replace("$", "").replace("€", "").replace(",", "").replace(", ", "").replace("\\", "")
            pattern = '-?\d*\.?\d+'
            pred = re.findall(pattern, pred_str)
            if (len(pred) >= 1):
                pred = pred[-1]
            else:
                pred = ''
    pred = pred.split("\n")[0]
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred)
    return pred


def extract_answer_str_by_answer_pattern(pred_str, answer_pattern):
    pred_result = pred_str
    if "json" in answer_pattern:
        first_dict_list = re.findall("\{.*?}", pred_str, re.DOTALL)
        key = answer_pattern.replace("json:", "").strip()
        if len(first_dict_list) > 0 and key in first_dict_list[0]:
            pred_str = first_dict_list[0]
        try:
            result_dict = json.loads(pred_str)
            pred_result = str(result_dict[key])
        except (json.JSONDecodeError, KeyError, TypeError):
            key_list = re.findall(f"{key}.*?" + "}", pred_str, re.DOTALL)
            if len(key_list) > 0:
                pred_result = key_list[0]
            else:
                pred_result = pred_str
    elif "[[]]" == answer_pattern:
        p_list = re.findall('\[\[(.*?)]]', pred_str)
        if len(p_list) > 0:
            pred_result = p_list[-1]
    return pred_result



def extract_answer_by_question_source(pred_str, question_source):
    pred = ""
    if question_source in ["MATH"]:
        pred = extract_latex_answer(pred_str)
    elif question_source in ["GSM8K"]:
        if "he answer is " in pred_str:
            pred_str = pred_str.split("he answer is ")[1]
            pred_str = pred_str.replace("$", "").replace("€", "").replace(",", "").replace("£", "").replace(", ",
                                                                                                            "")
            p_list = re.findall('-?\d*\.?\d+', pred_str)
            if len(p_list) > 0:
                pred = p_list[0]
        else:
            pred_str = pred_str.replace("$", "").replace("€", "").replace(",", "").replace("£", "").replace(", ",
                                                                                                            "")
            p_list = re.findall('-?\d*\.?\d+', pred_str)
            if len(p_list) > 0:
                pred = p_list[-1]
    return pred