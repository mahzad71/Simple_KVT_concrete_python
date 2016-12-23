# -*- coding: utf-8 -*-
#
# Autogenerated by Thrift Compiler (0.9.3)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py:new_style,utf8strings,coding=utf-8
#

from thrift.Thrift import TType, TMessageType, TException, TApplicationException
import concrete.communication.ttypes
import concrete.services.ttypes


from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol, TProtocol
try:
  from thrift.protocol import fastbinary
except:
  fastbinary = None



class FetchResult(object):
  """
  Struct containing Communications from the FetchCommunicationService service.

  Attributes:
   - communications: a list of Communication objects that represent the results of the request
  """

  thrift_spec = (
    None, # 0
    (1, TType.LIST, 'communications', (TType.STRUCT,(concrete.communication.ttypes.Communication, concrete.communication.ttypes.Communication.thrift_spec)), None, ), # 1
  )

  def __init__(self, communications=None,):
    self.communications = communications

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 1:
        if ftype == TType.LIST:
          self.communications = []
          (_etype3, _size0) = iprot.readListBegin()
          for _i4 in xrange(_size0):
            _elem5 = concrete.communication.ttypes.Communication()
            _elem5.read(iprot)
            self.communications.append(_elem5)
          iprot.readListEnd()
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('FetchResult')
    if self.communications is not None:
      oprot.writeFieldBegin('communications', TType.LIST, 1)
      oprot.writeListBegin(TType.STRUCT, len(self.communications))
      for iter6 in self.communications:
        iter6.write(oprot)
      oprot.writeListEnd()
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    if self.communications is None:
      raise TProtocol.TProtocolException(message='Required field communications is unset!')
    return


  def __hash__(self):
    value = 17
    value = (value * 31) ^ hash(self.communications)
    return value

  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)

class FetchRequest(object):
  """
  Struct representing a request for FetchCommunicationService.

  Attributes:
   - communicationIds: a list of Communication IDs
   - auths: optional authorization mechanism
  """

  thrift_spec = (
    None, # 0
    (1, TType.LIST, 'communicationIds', (TType.STRING,None), None, ), # 1
    (2, TType.STRING, 'auths', None, None, ), # 2
  )

  def __init__(self, communicationIds=None, auths=None,):
    self.communicationIds = communicationIds
    self.auths = auths

  def read(self, iprot):
    if iprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None and fastbinary is not None:
      fastbinary.decode_binary(self, iprot.trans, (self.__class__, self.thrift_spec))
      return
    iprot.readStructBegin()
    while True:
      (fname, ftype, fid) = iprot.readFieldBegin()
      if ftype == TType.STOP:
        break
      if fid == 1:
        if ftype == TType.LIST:
          self.communicationIds = []
          (_etype10, _size7) = iprot.readListBegin()
          for _i11 in xrange(_size7):
            _elem12 = iprot.readString().decode('utf-8')
            self.communicationIds.append(_elem12)
          iprot.readListEnd()
        else:
          iprot.skip(ftype)
      elif fid == 2:
        if ftype == TType.STRING:
          self.auths = iprot.readString().decode('utf-8')
        else:
          iprot.skip(ftype)
      else:
        iprot.skip(ftype)
      iprot.readFieldEnd()
    iprot.readStructEnd()

  def write(self, oprot):
    if oprot.__class__ == TBinaryProtocol.TBinaryProtocolAccelerated and self.thrift_spec is not None and fastbinary is not None:
      oprot.trans.write(fastbinary.encode_binary(self, (self.__class__, self.thrift_spec)))
      return
    oprot.writeStructBegin('FetchRequest')
    if self.communicationIds is not None:
      oprot.writeFieldBegin('communicationIds', TType.LIST, 1)
      oprot.writeListBegin(TType.STRING, len(self.communicationIds))
      for iter13 in self.communicationIds:
        oprot.writeString(iter13.encode('utf-8'))
      oprot.writeListEnd()
      oprot.writeFieldEnd()
    if self.auths is not None:
      oprot.writeFieldBegin('auths', TType.STRING, 2)
      oprot.writeString(self.auths.encode('utf-8'))
      oprot.writeFieldEnd()
    oprot.writeFieldStop()
    oprot.writeStructEnd()

  def validate(self):
    if self.communicationIds is None:
      raise TProtocol.TProtocolException(message='Required field communicationIds is unset!')
    return


  def __hash__(self):
    value = 17
    value = (value * 31) ^ hash(self.communicationIds)
    value = (value * 31) ^ hash(self.auths)
    return value

  def __repr__(self):
    L = ['%s=%r' % (key, value)
      for key, value in self.__dict__.iteritems()]
    return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

  def __eq__(self, other):
    return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

  def __ne__(self, other):
    return not (self == other)