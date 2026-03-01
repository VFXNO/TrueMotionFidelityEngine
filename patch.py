import sys
import re

with open('src/interpolator.cpp', 'r') as f:
    content = f.read()

# Insert ofstream definition at the top
if '#include <fstream>' not in content:
    content = '#include <fstream>\n' + content

# Patch createTex function
new_createTex = '''auto createTex = [&](int w, int h, DXGI_FORMAT fmt,
                         Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                         Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& srv,
                         Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
      if (!m_device) {
          std::ofstream ofs("trace_interpolator.txt", std::ios::app);
          ofs << "[createTex] m_device is null! w=" << w << " h=" << h << std::endl;
          return;
      }
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width      = w;
      desc.Height     = h;
      desc.MipLevels  = 1;
      desc.ArraySize  = 1;
      desc.Format     = fmt;
      desc.SampleDesc.Count = 1;
      desc.Usage      = D3D11_USAGE_DEFAULT;
      desc.BindFlags  = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      
      std::ofstream ofs("trace_interpolator.txt", std::ios::app);
      ofs << "[createTex] Creating texture w=" << w << " h=" << h << " fmt=" << fmt << std::endl;
      
      if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) {
          ofs << "[createTex] CreateTexture2D failed" << std::endl;
          return;
      }
      if (FAILED(m_device->CreateShaderResourceView(tex.Get(), nullptr, &srv))) { 
          ofs << "[createTex] CreateSRV failed" << std::endl;
          tex.Reset(); 
          return; 
      }
      if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) { 
          ofs << "[createTex] CreateUAV failed" << std::endl;
          tex.Reset(); srv.Reset(); 
          return; 
      }
      ofs << "[createTex] Success" << std::endl;
    };'''

old_createTex = '''auto createTex = [&](int w, int h, DXGI_FORMAT fmt,
                         Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                         Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>& srv,
                         Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width      = w;
      desc.Height     = h;
      desc.MipLevels  = 1;
      desc.ArraySize  = 1;
      desc.Format     = fmt;
      desc.SampleDesc.Count = 1;
      desc.Usage      = D3D11_USAGE_DEFAULT;
      desc.BindFlags  = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) return;
      if (FAILED(m_device->CreateShaderResourceView(tex.Get(), nullptr, &srv)))  { tex.Reset(); return; }
      if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) { tex.Reset(); srv.Reset(); return; }
    };'''

if old_createTex in content:
    content = content.replace(old_createTex, new_createTex)
else:
    print('Could not find old_createTex')

# Patch createUavTex
new_createUavTex = '''auto createUavTex = [&](int w, int h, DXGI_FORMAT fmt,
                            Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                            Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
      std::ofstream ofs("trace_interpolator.txt", std::ios::app);
      ofs << "[createUavTex] Creating w=" << w << " h=" << h << std::endl;
      
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width = w;
      desc.Height = h;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = fmt;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) {
          ofs << "[createUavTex] CreateTexture2D failed" << std::endl;
          return;
      }
      if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) {
          ofs << "[createUavTex] CreateUAV failed" << std::endl;
          tex.Reset(); return; 
      }
      ofs << "[createUavTex] Success" << std::endl;
    };'''

old_createUavTex = '''auto createUavTex = [&](int w, int h, DXGI_FORMAT fmt,
                            Microsoft::WRL::ComPtr<ID3D11Texture2D>& tex,
                            Microsoft::WRL::ComPtr<ID3D11UnorderedAccessView>& uav) {
      D3D11_TEXTURE2D_DESC desc = {};
      desc.Width = w;
      desc.Height = h;
      desc.MipLevels = 1;
      desc.ArraySize = 1;
      desc.Format = fmt;
      desc.SampleDesc.Count = 1;
      desc.Usage = D3D11_USAGE_DEFAULT;
      desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
      if (FAILED(m_device->CreateTexture2D(&desc, nullptr, &tex))) return;
      if (FAILED(m_device->CreateUnorderedAccessView(tex.Get(), nullptr, &uav))) { tex.Reset(); return; }
    };'''

if old_createUavTex in content:
    content = content.replace(old_createUavTex, new_createUavTex)
else:
    print('Could not find old_createUavTex')

# Add trace at end of CreateResources:
if 'if (!m_outputTexture || !m_outputSrv || !m_outputUav ||' in content:
    content = content.replace('if (!m_outputTexture || !m_outputSrv || !m_outputUav ||', 'std::ofstream ofs("trace_interpolator.txt", std::ios::app); ofs << "[CreateResources] Checking valid resources" << std::endl; if (!m_outputTexture || !m_outputSrv || !m_outputUav ||')

if 'return true;\n}' in content:
    content = content.replace('return true;\n}', 'std::ofstream ofs_end("trace_interpolator.txt", std::ios::app); ofs_end << "[CreateResources] Returning true" << std::endl;\nreturn true;\n}')
    
if 'bool Interpolator::Execute(' in content:
    content = content.replace('bool Interpolator::Execute(', 'bool Interpolator::Execute(\n    std::ofstream ofs_exec("trace_interpolator.txt", std::ios::app);\n    ofs_exec << "[Execute] Started" << std::endl;\n')

with open('src/interpolator.cpp', 'w') as f:
    f.write(content)
print('Patched successfully')