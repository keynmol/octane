# coding: utf-8
lib = File.expand_path('../lib', __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require 'octane/version'

Gem::Specification.new do |spec|
  spec.name          = "octane"
  spec.version       = Octane::VERSION
  spec.authors       = ["Anton Sviridov"]
  spec.email         = ["keynmol@gmail.com"]
  spec.description   = %q{Easy to use feedforward neural network implementation with dropout capabilities}
  spec.summary       = %q{feedforward neural network implementation}
  spec.homepage      = ""
  spec.license       = "MIT"

  spec.files         = `git ls-files`.split($/)
  spec.executables   = spec.files.grep(%r{^bin/}) { |f| File.basename(f) }
  spec.test_files    = spec.files.grep(%r{^(test|spec|features)/})
  spec.require_paths = ["lib"]

  spec.add_development_dependency "bundler", "~> 1.3"
  spec.add_development_dependency "rake"
end
