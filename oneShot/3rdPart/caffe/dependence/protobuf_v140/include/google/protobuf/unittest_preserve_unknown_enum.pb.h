// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: google/protobuf/unittest_preserve_unknown_enum.proto

#ifndef PROTOBUF_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto__INCLUDED
#define PROTOBUF_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3002000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3002000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/generated_enum_reflection.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
namespace proto3_preserve_unknown_enum_unittest {
class MyMessage;
class MyMessageDefaultTypeInternal;
extern MyMessageDefaultTypeInternal _MyMessage_default_instance_;
class MyMessagePlusExtra;
class MyMessagePlusExtraDefaultTypeInternal;
extern MyMessagePlusExtraDefaultTypeInternal _MyMessagePlusExtra_default_instance_;
}  // namespace proto3_preserve_unknown_enum_unittest

namespace proto3_preserve_unknown_enum_unittest {

namespace protobuf_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto {
// Internal implementation detail -- do not call these.
struct TableStruct {
  static const ::google::protobuf::uint32 offsets[];
  static void InitDefaultsImpl();
  static void Shutdown();
};
void AddDescriptors();
void InitDefaults();
}  // namespace protobuf_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto

enum MyEnum {
  FOO = 0,
  BAR = 1,
  BAZ = 2,
  MyEnum_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  MyEnum_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool MyEnum_IsValid(int value);
const MyEnum MyEnum_MIN = FOO;
const MyEnum MyEnum_MAX = BAZ;
const int MyEnum_ARRAYSIZE = MyEnum_MAX + 1;

const ::google::protobuf::EnumDescriptor* MyEnum_descriptor();
inline const ::std::string& MyEnum_Name(MyEnum value) {
  return ::google::protobuf::internal::NameOfEnum(
    MyEnum_descriptor(), value);
}
inline bool MyEnum_Parse(
    const ::std::string& name, MyEnum* value) {
  return ::google::protobuf::internal::ParseNamedEnum<MyEnum>(
    MyEnum_descriptor(), name, value);
}
enum MyEnumPlusExtra {
  E_FOO = 0,
  E_BAR = 1,
  E_BAZ = 2,
  E_EXTRA = 3,
  MyEnumPlusExtra_INT_MIN_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32min,
  MyEnumPlusExtra_INT_MAX_SENTINEL_DO_NOT_USE_ = ::google::protobuf::kint32max
};
bool MyEnumPlusExtra_IsValid(int value);
const MyEnumPlusExtra MyEnumPlusExtra_MIN = E_FOO;
const MyEnumPlusExtra MyEnumPlusExtra_MAX = E_EXTRA;
const int MyEnumPlusExtra_ARRAYSIZE = MyEnumPlusExtra_MAX + 1;

const ::google::protobuf::EnumDescriptor* MyEnumPlusExtra_descriptor();
inline const ::std::string& MyEnumPlusExtra_Name(MyEnumPlusExtra value) {
  return ::google::protobuf::internal::NameOfEnum(
    MyEnumPlusExtra_descriptor(), value);
}
inline bool MyEnumPlusExtra_Parse(
    const ::std::string& name, MyEnumPlusExtra* value) {
  return ::google::protobuf::internal::ParseNamedEnum<MyEnumPlusExtra>(
    MyEnumPlusExtra_descriptor(), name, value);
}
// ===================================================================

class MyMessage : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:proto3_preserve_unknown_enum_unittest.MyMessage) */ {
 public:
  MyMessage();
  virtual ~MyMessage();

  MyMessage(const MyMessage& from);

  inline MyMessage& operator=(const MyMessage& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const MyMessage& default_instance();

  enum OCase {
    kOneofE1 = 5,
    kOneofE2 = 6,
    O_NOT_SET = 0,
  };

  static inline const MyMessage* internal_default_instance() {
    return reinterpret_cast<const MyMessage*>(
               &_MyMessage_default_instance_);
  }

  void Swap(MyMessage* other);

  // implements Message ----------------------------------------------

  inline MyMessage* New() const PROTOBUF_FINAL { return New(NULL); }

  MyMessage* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const MyMessage& from);
  void MergeFrom(const MyMessage& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(MyMessage* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .proto3_preserve_unknown_enum_unittest.MyEnum repeated_e = 2;
  int repeated_e_size() const;
  void clear_repeated_e();
  static const int kRepeatedEFieldNumber = 2;
  ::proto3_preserve_unknown_enum_unittest::MyEnum repeated_e(int index) const;
  void set_repeated_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnum value);
  void add_repeated_e(::proto3_preserve_unknown_enum_unittest::MyEnum value);
  const ::google::protobuf::RepeatedField<int>& repeated_e() const;
  ::google::protobuf::RepeatedField<int>* mutable_repeated_e();

  // repeated .proto3_preserve_unknown_enum_unittest.MyEnum repeated_packed_e = 3 [packed = true];
  int repeated_packed_e_size() const;
  void clear_repeated_packed_e();
  static const int kRepeatedPackedEFieldNumber = 3;
  ::proto3_preserve_unknown_enum_unittest::MyEnum repeated_packed_e(int index) const;
  void set_repeated_packed_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnum value);
  void add_repeated_packed_e(::proto3_preserve_unknown_enum_unittest::MyEnum value);
  const ::google::protobuf::RepeatedField<int>& repeated_packed_e() const;
  ::google::protobuf::RepeatedField<int>* mutable_repeated_packed_e();

  // repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_packed_unexpected_e = 4;
  int repeated_packed_unexpected_e_size() const;
  void clear_repeated_packed_unexpected_e();
  static const int kRepeatedPackedUnexpectedEFieldNumber = 4;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra repeated_packed_unexpected_e(int index) const;
  void set_repeated_packed_unexpected_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  void add_repeated_packed_unexpected_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  const ::google::protobuf::RepeatedField<int>& repeated_packed_unexpected_e() const;
  ::google::protobuf::RepeatedField<int>* mutable_repeated_packed_unexpected_e();

  // .proto3_preserve_unknown_enum_unittest.MyEnum e = 1;
  void clear_e();
  static const int kEFieldNumber = 1;
  ::proto3_preserve_unknown_enum_unittest::MyEnum e() const;
  void set_e(::proto3_preserve_unknown_enum_unittest::MyEnum value);

  // .proto3_preserve_unknown_enum_unittest.MyEnum oneof_e_1 = 5;
  private:
  bool has_oneof_e_1() const;
  public:
  void clear_oneof_e_1();
  static const int kOneofE1FieldNumber = 5;
  ::proto3_preserve_unknown_enum_unittest::MyEnum oneof_e_1() const;
  void set_oneof_e_1(::proto3_preserve_unknown_enum_unittest::MyEnum value);

  // .proto3_preserve_unknown_enum_unittest.MyEnum oneof_e_2 = 6;
  private:
  bool has_oneof_e_2() const;
  public:
  void clear_oneof_e_2();
  static const int kOneofE2FieldNumber = 6;
  ::proto3_preserve_unknown_enum_unittest::MyEnum oneof_e_2() const;
  void set_oneof_e_2(::proto3_preserve_unknown_enum_unittest::MyEnum value);

  OCase o_case() const;
  // @@protoc_insertion_point(class_scope:proto3_preserve_unknown_enum_unittest.MyMessage)
 private:
  void set_has_oneof_e_1();
  void set_has_oneof_e_2();

  inline bool has_o() const;
  void clear_o();
  inline void clear_has_o();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField<int> repeated_e_;
  mutable int _repeated_e_cached_byte_size_;
  ::google::protobuf::RepeatedField<int> repeated_packed_e_;
  mutable int _repeated_packed_e_cached_byte_size_;
  ::google::protobuf::RepeatedField<int> repeated_packed_unexpected_e_;
  mutable int _repeated_packed_unexpected_e_cached_byte_size_;
  int e_;
  union OUnion {
    OUnion() {}
    int oneof_e_1_;
    int oneof_e_2_;
  } o_;
  mutable int _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct  protobuf_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto::TableStruct;
};
// -------------------------------------------------------------------

class MyMessagePlusExtra : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra) */ {
 public:
  MyMessagePlusExtra();
  virtual ~MyMessagePlusExtra();

  MyMessagePlusExtra(const MyMessagePlusExtra& from);

  inline MyMessagePlusExtra& operator=(const MyMessagePlusExtra& from) {
    CopyFrom(from);
    return *this;
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const MyMessagePlusExtra& default_instance();

  enum OCase {
    kOneofE1 = 5,
    kOneofE2 = 6,
    O_NOT_SET = 0,
  };

  static inline const MyMessagePlusExtra* internal_default_instance() {
    return reinterpret_cast<const MyMessagePlusExtra*>(
               &_MyMessagePlusExtra_default_instance_);
  }

  void Swap(MyMessagePlusExtra* other);

  // implements Message ----------------------------------------------

  inline MyMessagePlusExtra* New() const PROTOBUF_FINAL { return New(NULL); }

  MyMessagePlusExtra* New(::google::protobuf::Arena* arena) const PROTOBUF_FINAL;
  void CopyFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void MergeFrom(const ::google::protobuf::Message& from) PROTOBUF_FINAL;
  void CopyFrom(const MyMessagePlusExtra& from);
  void MergeFrom(const MyMessagePlusExtra& from);
  void Clear() PROTOBUF_FINAL;
  bool IsInitialized() const PROTOBUF_FINAL;

  size_t ByteSizeLong() const PROTOBUF_FINAL;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input) PROTOBUF_FINAL;
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* target) const PROTOBUF_FINAL;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output)
      const PROTOBUF_FINAL {
    return InternalSerializeWithCachedSizesToArray(
        ::google::protobuf::io::CodedOutputStream::IsDefaultSerializationDeterministic(), output);
  }
  int GetCachedSize() const PROTOBUF_FINAL { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const PROTOBUF_FINAL;
  void InternalSwap(MyMessagePlusExtra* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return NULL;
  }
  inline void* MaybeArenaPtr() const {
    return NULL;
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const PROTOBUF_FINAL;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_e = 2;
  int repeated_e_size() const;
  void clear_repeated_e();
  static const int kRepeatedEFieldNumber = 2;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra repeated_e(int index) const;
  void set_repeated_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  void add_repeated_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  const ::google::protobuf::RepeatedField<int>& repeated_e() const;
  ::google::protobuf::RepeatedField<int>* mutable_repeated_e();

  // repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_packed_e = 3 [packed = true];
  int repeated_packed_e_size() const;
  void clear_repeated_packed_e();
  static const int kRepeatedPackedEFieldNumber = 3;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra repeated_packed_e(int index) const;
  void set_repeated_packed_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  void add_repeated_packed_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  const ::google::protobuf::RepeatedField<int>& repeated_packed_e() const;
  ::google::protobuf::RepeatedField<int>* mutable_repeated_packed_e();

  // repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_packed_unexpected_e = 4 [packed = true];
  int repeated_packed_unexpected_e_size() const;
  void clear_repeated_packed_unexpected_e();
  static const int kRepeatedPackedUnexpectedEFieldNumber = 4;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra repeated_packed_unexpected_e(int index) const;
  void set_repeated_packed_unexpected_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  void add_repeated_packed_unexpected_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);
  const ::google::protobuf::RepeatedField<int>& repeated_packed_unexpected_e() const;
  ::google::protobuf::RepeatedField<int>* mutable_repeated_packed_unexpected_e();

  // .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra e = 1;
  void clear_e();
  static const int kEFieldNumber = 1;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra e() const;
  void set_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);

  // .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra oneof_e_1 = 5;
  private:
  bool has_oneof_e_1() const;
  public:
  void clear_oneof_e_1();
  static const int kOneofE1FieldNumber = 5;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra oneof_e_1() const;
  void set_oneof_e_1(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);

  // .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra oneof_e_2 = 6;
  private:
  bool has_oneof_e_2() const;
  public:
  void clear_oneof_e_2();
  static const int kOneofE2FieldNumber = 6;
  ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra oneof_e_2() const;
  void set_oneof_e_2(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value);

  OCase o_case() const;
  // @@protoc_insertion_point(class_scope:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra)
 private:
  void set_has_oneof_e_1();
  void set_has_oneof_e_2();

  inline bool has_o() const;
  void clear_o();
  inline void clear_has_o();

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::RepeatedField<int> repeated_e_;
  mutable int _repeated_e_cached_byte_size_;
  ::google::protobuf::RepeatedField<int> repeated_packed_e_;
  mutable int _repeated_packed_e_cached_byte_size_;
  ::google::protobuf::RepeatedField<int> repeated_packed_unexpected_e_;
  mutable int _repeated_packed_unexpected_e_cached_byte_size_;
  int e_;
  union OUnion {
    OUnion() {}
    int oneof_e_1_;
    int oneof_e_2_;
  } o_;
  mutable int _cached_size_;
  ::google::protobuf::uint32 _oneof_case_[1];

  friend struct  protobuf_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto::TableStruct;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// MyMessage

// .proto3_preserve_unknown_enum_unittest.MyEnum e = 1;
inline void MyMessage::clear_e() {
  e_ = 0;
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnum MyMessage::e() const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessage.e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(e_);
}
inline void MyMessage::set_e(::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  
  e_ = value;
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessage.e)
}

// repeated .proto3_preserve_unknown_enum_unittest.MyEnum repeated_e = 2;
inline int MyMessage::repeated_e_size() const {
  return repeated_e_.size();
}
inline void MyMessage::clear_repeated_e() {
  repeated_e_.Clear();
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnum MyMessage::repeated_e(int index) const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(repeated_e_.Get(index));
}
inline void MyMessage::set_repeated_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  repeated_e_.Set(index, value);
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_e)
}
inline void MyMessage::add_repeated_e(::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  repeated_e_.Add(value);
  // @@protoc_insertion_point(field_add:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_e)
}
inline const ::google::protobuf::RepeatedField<int>&
MyMessage::repeated_e() const {
  // @@protoc_insertion_point(field_list:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_e)
  return repeated_e_;
}
inline ::google::protobuf::RepeatedField<int>*
MyMessage::mutable_repeated_e() {
  // @@protoc_insertion_point(field_mutable_list:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_e)
  return &repeated_e_;
}

// repeated .proto3_preserve_unknown_enum_unittest.MyEnum repeated_packed_e = 3 [packed = true];
inline int MyMessage::repeated_packed_e_size() const {
  return repeated_packed_e_.size();
}
inline void MyMessage::clear_repeated_packed_e() {
  repeated_packed_e_.Clear();
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnum MyMessage::repeated_packed_e(int index) const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(repeated_packed_e_.Get(index));
}
inline void MyMessage::set_repeated_packed_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  repeated_packed_e_.Set(index, value);
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_e)
}
inline void MyMessage::add_repeated_packed_e(::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  repeated_packed_e_.Add(value);
  // @@protoc_insertion_point(field_add:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_e)
}
inline const ::google::protobuf::RepeatedField<int>&
MyMessage::repeated_packed_e() const {
  // @@protoc_insertion_point(field_list:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_e)
  return repeated_packed_e_;
}
inline ::google::protobuf::RepeatedField<int>*
MyMessage::mutable_repeated_packed_e() {
  // @@protoc_insertion_point(field_mutable_list:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_e)
  return &repeated_packed_e_;
}

// repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_packed_unexpected_e = 4;
inline int MyMessage::repeated_packed_unexpected_e_size() const {
  return repeated_packed_unexpected_e_.size();
}
inline void MyMessage::clear_repeated_packed_unexpected_e() {
  repeated_packed_unexpected_e_.Clear();
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessage::repeated_packed_unexpected_e(int index) const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_unexpected_e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(repeated_packed_unexpected_e_.Get(index));
}
inline void MyMessage::set_repeated_packed_unexpected_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_packed_unexpected_e_.Set(index, value);
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_unexpected_e)
}
inline void MyMessage::add_repeated_packed_unexpected_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_packed_unexpected_e_.Add(value);
  // @@protoc_insertion_point(field_add:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_unexpected_e)
}
inline const ::google::protobuf::RepeatedField<int>&
MyMessage::repeated_packed_unexpected_e() const {
  // @@protoc_insertion_point(field_list:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_unexpected_e)
  return repeated_packed_unexpected_e_;
}
inline ::google::protobuf::RepeatedField<int>*
MyMessage::mutable_repeated_packed_unexpected_e() {
  // @@protoc_insertion_point(field_mutable_list:proto3_preserve_unknown_enum_unittest.MyMessage.repeated_packed_unexpected_e)
  return &repeated_packed_unexpected_e_;
}

// .proto3_preserve_unknown_enum_unittest.MyEnum oneof_e_1 = 5;
inline bool MyMessage::has_oneof_e_1() const {
  return o_case() == kOneofE1;
}
inline void MyMessage::set_has_oneof_e_1() {
  _oneof_case_[0] = kOneofE1;
}
inline void MyMessage::clear_oneof_e_1() {
  if (has_oneof_e_1()) {
    o_.oneof_e_1_ = 0;
    clear_has_o();
  }
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnum MyMessage::oneof_e_1() const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessage.oneof_e_1)
  if (has_oneof_e_1()) {
    return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(o_.oneof_e_1_);
  }
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(0);
}
inline void MyMessage::set_oneof_e_1(::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  if (!has_oneof_e_1()) {
    clear_o();
    set_has_oneof_e_1();
  }
  o_.oneof_e_1_ = value;
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessage.oneof_e_1)
}

// .proto3_preserve_unknown_enum_unittest.MyEnum oneof_e_2 = 6;
inline bool MyMessage::has_oneof_e_2() const {
  return o_case() == kOneofE2;
}
inline void MyMessage::set_has_oneof_e_2() {
  _oneof_case_[0] = kOneofE2;
}
inline void MyMessage::clear_oneof_e_2() {
  if (has_oneof_e_2()) {
    o_.oneof_e_2_ = 0;
    clear_has_o();
  }
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnum MyMessage::oneof_e_2() const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessage.oneof_e_2)
  if (has_oneof_e_2()) {
    return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(o_.oneof_e_2_);
  }
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnum >(0);
}
inline void MyMessage::set_oneof_e_2(::proto3_preserve_unknown_enum_unittest::MyEnum value) {
  if (!has_oneof_e_2()) {
    clear_o();
    set_has_oneof_e_2();
  }
  o_.oneof_e_2_ = value;
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessage.oneof_e_2)
}

inline bool MyMessage::has_o() const {
  return o_case() != O_NOT_SET;
}
inline void MyMessage::clear_has_o() {
  _oneof_case_[0] = O_NOT_SET;
}
inline MyMessage::OCase MyMessage::o_case() const {
  return MyMessage::OCase(_oneof_case_[0]);
}
// -------------------------------------------------------------------

// MyMessagePlusExtra

// .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra e = 1;
inline void MyMessagePlusExtra::clear_e() {
  e_ = 0;
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessagePlusExtra::e() const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(e_);
}
inline void MyMessagePlusExtra::set_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  
  e_ = value;
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.e)
}

// repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_e = 2;
inline int MyMessagePlusExtra::repeated_e_size() const {
  return repeated_e_.size();
}
inline void MyMessagePlusExtra::clear_repeated_e() {
  repeated_e_.Clear();
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessagePlusExtra::repeated_e(int index) const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(repeated_e_.Get(index));
}
inline void MyMessagePlusExtra::set_repeated_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_e_.Set(index, value);
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_e)
}
inline void MyMessagePlusExtra::add_repeated_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_e_.Add(value);
  // @@protoc_insertion_point(field_add:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_e)
}
inline const ::google::protobuf::RepeatedField<int>&
MyMessagePlusExtra::repeated_e() const {
  // @@protoc_insertion_point(field_list:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_e)
  return repeated_e_;
}
inline ::google::protobuf::RepeatedField<int>*
MyMessagePlusExtra::mutable_repeated_e() {
  // @@protoc_insertion_point(field_mutable_list:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_e)
  return &repeated_e_;
}

// repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_packed_e = 3 [packed = true];
inline int MyMessagePlusExtra::repeated_packed_e_size() const {
  return repeated_packed_e_.size();
}
inline void MyMessagePlusExtra::clear_repeated_packed_e() {
  repeated_packed_e_.Clear();
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessagePlusExtra::repeated_packed_e(int index) const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(repeated_packed_e_.Get(index));
}
inline void MyMessagePlusExtra::set_repeated_packed_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_packed_e_.Set(index, value);
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_e)
}
inline void MyMessagePlusExtra::add_repeated_packed_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_packed_e_.Add(value);
  // @@protoc_insertion_point(field_add:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_e)
}
inline const ::google::protobuf::RepeatedField<int>&
MyMessagePlusExtra::repeated_packed_e() const {
  // @@protoc_insertion_point(field_list:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_e)
  return repeated_packed_e_;
}
inline ::google::protobuf::RepeatedField<int>*
MyMessagePlusExtra::mutable_repeated_packed_e() {
  // @@protoc_insertion_point(field_mutable_list:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_e)
  return &repeated_packed_e_;
}

// repeated .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra repeated_packed_unexpected_e = 4 [packed = true];
inline int MyMessagePlusExtra::repeated_packed_unexpected_e_size() const {
  return repeated_packed_unexpected_e_.size();
}
inline void MyMessagePlusExtra::clear_repeated_packed_unexpected_e() {
  repeated_packed_unexpected_e_.Clear();
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessagePlusExtra::repeated_packed_unexpected_e(int index) const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_unexpected_e)
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(repeated_packed_unexpected_e_.Get(index));
}
inline void MyMessagePlusExtra::set_repeated_packed_unexpected_e(int index, ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_packed_unexpected_e_.Set(index, value);
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_unexpected_e)
}
inline void MyMessagePlusExtra::add_repeated_packed_unexpected_e(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  repeated_packed_unexpected_e_.Add(value);
  // @@protoc_insertion_point(field_add:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_unexpected_e)
}
inline const ::google::protobuf::RepeatedField<int>&
MyMessagePlusExtra::repeated_packed_unexpected_e() const {
  // @@protoc_insertion_point(field_list:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_unexpected_e)
  return repeated_packed_unexpected_e_;
}
inline ::google::protobuf::RepeatedField<int>*
MyMessagePlusExtra::mutable_repeated_packed_unexpected_e() {
  // @@protoc_insertion_point(field_mutable_list:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.repeated_packed_unexpected_e)
  return &repeated_packed_unexpected_e_;
}

// .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra oneof_e_1 = 5;
inline bool MyMessagePlusExtra::has_oneof_e_1() const {
  return o_case() == kOneofE1;
}
inline void MyMessagePlusExtra::set_has_oneof_e_1() {
  _oneof_case_[0] = kOneofE1;
}
inline void MyMessagePlusExtra::clear_oneof_e_1() {
  if (has_oneof_e_1()) {
    o_.oneof_e_1_ = 0;
    clear_has_o();
  }
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessagePlusExtra::oneof_e_1() const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.oneof_e_1)
  if (has_oneof_e_1()) {
    return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(o_.oneof_e_1_);
  }
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(0);
}
inline void MyMessagePlusExtra::set_oneof_e_1(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  if (!has_oneof_e_1()) {
    clear_o();
    set_has_oneof_e_1();
  }
  o_.oneof_e_1_ = value;
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.oneof_e_1)
}

// .proto3_preserve_unknown_enum_unittest.MyEnumPlusExtra oneof_e_2 = 6;
inline bool MyMessagePlusExtra::has_oneof_e_2() const {
  return o_case() == kOneofE2;
}
inline void MyMessagePlusExtra::set_has_oneof_e_2() {
  _oneof_case_[0] = kOneofE2;
}
inline void MyMessagePlusExtra::clear_oneof_e_2() {
  if (has_oneof_e_2()) {
    o_.oneof_e_2_ = 0;
    clear_has_o();
  }
}
inline ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra MyMessagePlusExtra::oneof_e_2() const {
  // @@protoc_insertion_point(field_get:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.oneof_e_2)
  if (has_oneof_e_2()) {
    return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(o_.oneof_e_2_);
  }
  return static_cast< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra >(0);
}
inline void MyMessagePlusExtra::set_oneof_e_2(::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra value) {
  if (!has_oneof_e_2()) {
    clear_o();
    set_has_oneof_e_2();
  }
  o_.oneof_e_2_ = value;
  // @@protoc_insertion_point(field_set:proto3_preserve_unknown_enum_unittest.MyMessagePlusExtra.oneof_e_2)
}

inline bool MyMessagePlusExtra::has_o() const {
  return o_case() != O_NOT_SET;
}
inline void MyMessagePlusExtra::clear_has_o() {
  _oneof_case_[0] = O_NOT_SET;
}
inline MyMessagePlusExtra::OCase MyMessagePlusExtra::o_case() const {
  return MyMessagePlusExtra::OCase(_oneof_case_[0]);
}
#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)


}  // namespace proto3_preserve_unknown_enum_unittest

#ifndef SWIG
namespace google {
namespace protobuf {

template <> struct is_proto_enum< ::proto3_preserve_unknown_enum_unittest::MyEnum> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::proto3_preserve_unknown_enum_unittest::MyEnum>() {
  return ::proto3_preserve_unknown_enum_unittest::MyEnum_descriptor();
}
template <> struct is_proto_enum< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra> : ::google::protobuf::internal::true_type {};
template <>
inline const EnumDescriptor* GetEnumDescriptor< ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra>() {
  return ::proto3_preserve_unknown_enum_unittest::MyEnumPlusExtra_descriptor();
}

}  // namespace protobuf
}  // namespace google
#endif  // SWIG

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_google_2fprotobuf_2funittest_5fpreserve_5funknown_5fenum_2eproto__INCLUDED
