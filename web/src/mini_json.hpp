// Minimal JSON encoder/decoder sufficient for the solver interface.
//
// Supports: null, booleans, numbers (as doubles), strings, arrays, objects.
// Not a full JSON parser -- no unicode escapes beyond \" \\ \n \t, no
// comments, no streaming.  Thrown errors are std::runtime_error.
#pragma once

#include <cctype>
#include <cstddef>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace mjson {

class Value;
using Array  = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
 public:
  enum Kind { NIL, BOOL, NUM, STR, ARR, OBJ };

  Value() : kind_(NIL) {}
  Value(bool b) : kind_(BOOL), b_(b) {}
  Value(double d) : kind_(NUM), n_(d) {}
  Value(int i) : kind_(NUM), n_(static_cast<double>(i)) {}
  Value(const char* s) : kind_(STR), s_(s) {}
  Value(std::string s) : kind_(STR), s_(std::move(s)) {}
  Value(Array a) : kind_(ARR), a_(std::make_shared<Array>(std::move(a))) {}
  Value(Object o) : kind_(OBJ), o_(std::make_shared<Object>(std::move(o))) {}

  Kind kind() const { return kind_; }

  bool is_null() const { return kind_ == NIL; }
  bool is_bool() const { return kind_ == BOOL; }
  bool is_num()  const { return kind_ == NUM; }
  bool is_str()  const { return kind_ == STR; }
  bool is_arr()  const { return kind_ == ARR; }
  bool is_obj()  const { return kind_ == OBJ; }

  bool as_bool() const { check(BOOL, "bool"); return b_; }
  double as_num() const { check(NUM, "number"); return n_; }
  int as_int() const { return static_cast<int>(as_num()); }
  const std::string& as_str() const { check(STR, "string"); return s_; }
  Array& as_arr() { check(ARR, "array"); return *a_; }
  const Array& as_arr() const { check(ARR, "array"); return *a_; }
  Object& as_obj() { check(OBJ, "object"); return *o_; }
  const Object& as_obj() const { check(OBJ, "object"); return *o_; }

  bool contains(const std::string& k) const {
    return is_obj() && o_->count(k) > 0;
  }
  const Value& at(const std::string& k) const { return as_obj().at(k); }
  Value& operator[](const std::string& k) {
    if (!is_obj()) { kind_ = OBJ; o_ = std::make_shared<Object>(); }
    return (*o_)[k];
  }

  // Convenience getters with defaults.
  double value_num(const std::string& k, double d) const {
    if (!contains(k)) return d;
    return at(k).as_num();
  }
  int value_int(const std::string& k, int d) const {
    if (!contains(k)) return d;
    return at(k).as_int();
  }
  bool value_bool(const std::string& k, bool d) const {
    if (!contains(k)) return d;
    return at(k).as_bool();
  }

  std::string dump() const {
    std::ostringstream os; emit(os); return os.str();
  }

  static Value parse(const std::string& s) {
    Parser p(s);
    p.skip_ws();
    Value v = p.parse_value();
    p.skip_ws();
    if (p.pos_ != s.size()) p.fail("trailing content");
    return v;
  }

 private:
  void check(Kind want, const char* name) const {
    if (kind_ != want) throw std::runtime_error(std::string("expected ") + name);
  }

  void emit(std::ostringstream& os) const {
    switch (kind_) {
      case NIL:  os << "null"; break;
      case BOOL: os << (b_ ? "true" : "false"); break;
      case NUM:
        if (std::isnan(n_)) os << "null";
        else if (std::isinf(n_)) os << (n_ < 0 ? "-1e400" : "1e400");
        else {
          // Prefer integer formatting when the value is integral.
          double r = std::round(n_);
          if (std::abs(n_ - r) < 1e-12 && std::abs(r) < 1e15) {
            os << static_cast<long long>(r);
          } else {
            char buf[40];
            std::snprintf(buf, sizeof(buf), "%.17g", n_);
            os << buf;
          }
        }
        break;
      case STR:
        os << '"';
        for (char c : s_) {
          switch (c) {
            case '"':  os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\t': os << "\\t"; break;
            case '\r': os << "\\r"; break;
            default: os << c;
          }
        }
        os << '"';
        break;
      case ARR: {
        os << '[';
        bool first = true;
        for (const auto& v : *a_) { if (!first) os << ','; first = false; v.emit(os); }
        os << ']';
        break;
      }
      case OBJ: {
        os << '{';
        bool first = true;
        for (const auto& [k, v] : *o_) {
          if (!first) os << ','; first = false;
          os << '"' << k << "\":"; v.emit(os);
        }
        os << '}';
        break;
      }
    }
  }

  class Parser {
   public:
    explicit Parser(const std::string& s) : src_(s) {}
    void skip_ws() {
      while (pos_ < src_.size() && std::isspace(static_cast<unsigned char>(src_[pos_]))) ++pos_;
    }
    void fail(const std::string& msg) const {
      throw std::runtime_error("json parse error at pos " + std::to_string(pos_)
                               + ": " + msg);
    }
    Value parse_value() {
      skip_ws();
      if (pos_ >= src_.size()) fail("unexpected end");
      char c = src_[pos_];
      if (c == '{') return parse_object();
      if (c == '[') return parse_array();
      if (c == '"') return parse_string();
      if (c == 't' || c == 'f') return parse_bool();
      if (c == 'n') return parse_null();
      return parse_number();
    }
    Value parse_object() {
      ++pos_; skip_ws();
      Object o;
      if (pos_ < src_.size() && src_[pos_] == '}') { ++pos_; return Value(std::move(o)); }
      while (true) {
        skip_ws();
        if (pos_ >= src_.size() || src_[pos_] != '"') fail("expected string key");
        std::string key = parse_string().as_str();
        skip_ws();
        if (pos_ >= src_.size() || src_[pos_] != ':') fail("expected ':'");
        ++pos_;
        Value v = parse_value();
        o.emplace(std::move(key), std::move(v));
        skip_ws();
        if (pos_ >= src_.size()) fail("unterminated object");
        if (src_[pos_] == ',') { ++pos_; continue; }
        if (src_[pos_] == '}') { ++pos_; break; }
        fail("expected ',' or '}'");
      }
      return Value(std::move(o));
    }
    Value parse_array() {
      ++pos_; skip_ws();
      Array a;
      if (pos_ < src_.size() && src_[pos_] == ']') { ++pos_; return Value(std::move(a)); }
      while (true) {
        a.push_back(parse_value());
        skip_ws();
        if (pos_ >= src_.size()) fail("unterminated array");
        if (src_[pos_] == ',') { ++pos_; continue; }
        if (src_[pos_] == ']') { ++pos_; break; }
        fail("expected ',' or ']'");
      }
      return Value(std::move(a));
    }
    Value parse_string() {
      ++pos_;
      std::string out;
      while (pos_ < src_.size() && src_[pos_] != '"') {
        char c = src_[pos_++];
        if (c == '\\' && pos_ < src_.size()) {
          char e = src_[pos_++];
          switch (e) {
            case '"':  out += '"'; break;
            case '\\': out += '\\'; break;
            case '/':  out += '/'; break;
            case 'n':  out += '\n'; break;
            case 't':  out += '\t'; break;
            case 'r':  out += '\r'; break;
            default:   out += e;
          }
        } else {
          out += c;
        }
      }
      if (pos_ >= src_.size()) fail("unterminated string");
      ++pos_; // closing quote
      return Value(std::move(out));
    }
    Value parse_bool() {
      if (src_.compare(pos_, 4, "true") == 0)  { pos_ += 4; return Value(true); }
      if (src_.compare(pos_, 5, "false") == 0) { pos_ += 5; return Value(false); }
      fail("expected bool");
      return Value();
    }
    Value parse_null() {
      if (src_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Value(); }
      fail("expected null");
      return Value();
    }
    Value parse_number() {
      size_t start = pos_;
      if (src_[pos_] == '-' || src_[pos_] == '+') ++pos_;
      while (pos_ < src_.size() &&
             (std::isdigit(static_cast<unsigned char>(src_[pos_])) ||
              src_[pos_] == '.' || src_[pos_] == 'e' || src_[pos_] == 'E' ||
              src_[pos_] == '+' || src_[pos_] == '-')) ++pos_;
      if (pos_ == start) fail("expected number");
      return Value(std::stod(src_.substr(start, pos_ - start)));
    }
    const std::string& src_;
    size_t pos_ = 0;
  };

  Kind kind_;
  bool b_ = false;
  double n_ = 0.0;
  std::string s_;
  std::shared_ptr<Array> a_;
  std::shared_ptr<Object> o_;
};

}  // namespace mjson
