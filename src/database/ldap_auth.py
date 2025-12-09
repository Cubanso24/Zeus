"""LDAP Authentication module for Zeus.

Supports hybrid authentication with both LDAP and local accounts.
Compatible with Active Directory, OpenLDAP, and other LDAP servers.
Configuration can be loaded from database or environment variables.
"""

import os
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Try to import ldap3 (pure Python, no system dependencies)
try:
    from ldap3 import Server, Connection, ALL, SUBTREE, NTLM
    from ldap3.core.exceptions import LDAPException, LDAPBindError, LDAPSocketOpenError
    LDAP_AVAILABLE = True
except ImportError:
    LDAP_AVAILABLE = False
    logger.warning("ldap3 not installed. LDAP authentication will be disabled.")


@dataclass
class LDAPConfig:
    """LDAP Configuration settings."""
    enabled: bool = False
    server_url: str = ""  # e.g., "ldap://ldap.example.com:389" or "ldaps://ldap.example.com:636"
    use_ssl: bool = False  # Use LDAPS
    use_tls: bool = True   # Use STARTTLS on non-SSL connection

    # Bind credentials (for searching users)
    bind_dn: str = ""      # e.g., "cn=readonly,dc=example,dc=com" or "readonly@example.com"
    bind_password: str = ""

    # Search configuration
    base_dn: str = ""      # e.g., "dc=example,dc=com"
    user_search_filter: str = "(|(uid={username})(sAMAccountName={username})(mail={username}))"
    user_search_scope: str = "SUBTREE"

    # Attribute mapping (configurable for different LDAP servers)
    username_attribute: str = "uid"           # OpenLDAP: uid, AD: sAMAccountName
    email_attribute: str = "mail"
    fullname_attribute: str = "cn"            # Common name

    # Connection settings
    connect_timeout: int = 10
    receive_timeout: int = 10

    # Authentication method
    auth_method: str = "SIMPLE"  # SIMPLE or NTLM (for AD)

    @classmethod
    def from_env(cls) -> "LDAPConfig":
        """Load LDAP configuration from environment variables."""
        return cls(
            enabled=os.getenv("LDAP_ENABLED", "false").lower() == "true",
            server_url=os.getenv("LDAP_SERVER_URL", ""),
            use_ssl=os.getenv("LDAP_USE_SSL", "false").lower() == "true",
            use_tls=os.getenv("LDAP_USE_TLS", "true").lower() == "true",
            bind_dn=os.getenv("LDAP_BIND_DN", ""),
            bind_password=os.getenv("LDAP_BIND_PASSWORD", ""),
            base_dn=os.getenv("LDAP_BASE_DN", ""),
            user_search_filter=os.getenv(
                "LDAP_USER_SEARCH_FILTER",
                "(|(uid={username})(sAMAccountName={username})(mail={username}))"
            ),
            user_search_scope=os.getenv("LDAP_USER_SEARCH_SCOPE", "SUBTREE"),
            username_attribute=os.getenv("LDAP_USERNAME_ATTR", "uid"),
            email_attribute=os.getenv("LDAP_EMAIL_ATTR", "mail"),
            fullname_attribute=os.getenv("LDAP_FULLNAME_ATTR", "cn"),
            connect_timeout=int(os.getenv("LDAP_CONNECT_TIMEOUT", "10")),
            receive_timeout=int(os.getenv("LDAP_RECEIVE_TIMEOUT", "10")),
            auth_method=os.getenv("LDAP_AUTH_METHOD", "SIMPLE"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LDAPConfig":
        """Load LDAP configuration from a dictionary."""
        return cls(
            enabled=data.get("enabled", False),
            server_url=data.get("server_url", ""),
            use_ssl=data.get("use_ssl", False),
            use_tls=data.get("use_tls", True),
            bind_dn=data.get("bind_dn", ""),
            bind_password=data.get("bind_password", ""),
            base_dn=data.get("base_dn", ""),
            user_search_filter=data.get(
                "user_search_filter",
                "(|(uid={username})(sAMAccountName={username})(mail={username}))"
            ),
            user_search_scope=data.get("user_search_scope", "SUBTREE"),
            username_attribute=data.get("username_attribute", "uid"),
            email_attribute=data.get("email_attribute", "mail"),
            fullname_attribute=data.get("fullname_attribute", "cn"),
            connect_timeout=int(data.get("connect_timeout", 10)),
            receive_timeout=int(data.get("receive_timeout", 10)),
            auth_method=data.get("auth_method", "SIMPLE"),
        )

    def to_dict(self, include_password: bool = False) -> Dict[str, Any]:
        """Convert configuration to dictionary (optionally excluding password)."""
        data = asdict(self)
        if not include_password:
            data["bind_password"] = "********" if self.bind_password else ""
        return data


class LDAPAuthenticator:
    """LDAP Authentication handler."""

    def __init__(self, config: Optional[LDAPConfig] = None):
        """Initialize LDAP authenticator with configuration."""
        self.config = config or LDAPConfig.from_env()
        self._server: Optional[Server] = None

    @property
    def is_available(self) -> bool:
        """Check if LDAP authentication is available and enabled."""
        return LDAP_AVAILABLE and self.config.enabled and bool(self.config.server_url)

    def _get_server(self) -> "Server":
        """Get or create LDAP server connection."""
        if not LDAP_AVAILABLE:
            raise RuntimeError("ldap3 library not installed")

        if self._server is None:
            self._server = Server(
                self.config.server_url,
                use_ssl=self.config.use_ssl,
                get_info=ALL,
                connect_timeout=self.config.connect_timeout
            )
        return self._server

    def _create_connection(self, user_dn: str = None, password: str = None) -> "Connection":
        """Create an LDAP connection."""
        server = self._get_server()

        # Use bind credentials if no user credentials provided
        if user_dn is None:
            user_dn = self.config.bind_dn
            password = self.config.bind_password

        # Determine authentication method
        if self.config.auth_method.upper() == "NTLM":
            conn = Connection(
                server,
                user=user_dn,
                password=password,
                authentication=NTLM,
                auto_bind=False,
                receive_timeout=self.config.receive_timeout
            )
        else:
            conn = Connection(
                server,
                user=user_dn,
                password=password,
                auto_bind=False,
                receive_timeout=self.config.receive_timeout
            )

        return conn

    def _bind_connection(self, conn: "Connection") -> bool:
        """Bind connection with optional STARTTLS."""
        try:
            if not conn.bind():
                return False

            # Start TLS if configured and not using SSL
            if self.config.use_tls and not self.config.use_ssl:
                conn.start_tls()

            return True
        except LDAPException as e:
            logger.error(f"LDAP bind failed: {e}")
            return False

    def search_user(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Search for a user in LDAP by username.

        Args:
            username: Username to search for

        Returns:
            Dictionary with user info or None if not found
        """
        if not self.is_available:
            return None

        try:
            conn = self._create_connection()
            if not self._bind_connection(conn):
                logger.error("Failed to bind to LDAP server for user search")
                return None

            # Build search filter
            search_filter = self.config.user_search_filter.format(username=username)

            # Determine search scope
            scope = SUBTREE
            if self.config.user_search_scope.upper() == "ONELEVEL":
                from ldap3 import LEVEL
                scope = LEVEL
            elif self.config.user_search_scope.upper() == "BASE":
                from ldap3 import BASE
                scope = BASE

            # Search for user
            conn.search(
                search_base=self.config.base_dn,
                search_filter=search_filter,
                search_scope=scope,
                attributes=[
                    self.config.username_attribute,
                    self.config.email_attribute,
                    self.config.fullname_attribute,
                ]
            )

            if not conn.entries:
                logger.debug(f"User not found in LDAP: {username}")
                return None

            # Get first matching entry
            entry = conn.entries[0]

            # Extract attributes
            user_info = {
                "dn": entry.entry_dn,
                "username": self._get_attribute(entry, self.config.username_attribute, username),
                "email": self._get_attribute(entry, self.config.email_attribute, f"{username}@ldap.local"),
                "full_name": self._get_attribute(entry, self.config.fullname_attribute, username),
            }

            conn.unbind()
            return user_info

        except LDAPSocketOpenError as e:
            logger.error(f"Cannot connect to LDAP server: {e}")
            return None
        except LDAPException as e:
            logger.error(f"LDAP search error: {e}")
            return None

    def _get_attribute(self, entry, attr_name: str, default: str = "") -> str:
        """Safely get an attribute value from an LDAP entry."""
        try:
            attr = getattr(entry, attr_name, None)
            if attr and attr.value:
                # Handle multi-valued attributes
                if isinstance(attr.value, list):
                    return str(attr.value[0]) if attr.value else default
                return str(attr.value)
        except Exception:
            pass
        return default

    def authenticate(self, username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Authenticate a user against LDAP.

        Args:
            username: Username to authenticate
            password: Password to verify

        Returns:
            Tuple of (success: bool, user_info: dict or None)
        """
        if not self.is_available:
            return False, None

        if not password:
            logger.warning("Empty password provided for LDAP authentication")
            return False, None

        try:
            # First, search for the user to get their DN and info
            user_info = self.search_user(username)
            if not user_info:
                logger.debug(f"User not found in LDAP: {username}")
                return False, None

            user_dn = user_info["dn"]

            # Now try to bind as the user
            conn = self._create_connection(user_dn, password)

            try:
                if not conn.bind():
                    logger.debug(f"LDAP bind failed for user: {username}")
                    return False, None

                # Bind successful - user is authenticated
                conn.unbind()
                logger.info(f"LDAP authentication successful for user: {username}")
                return True, user_info

            except LDAPBindError as e:
                logger.debug(f"LDAP bind error for {username}: {e}")
                return False, None

        except LDAPSocketOpenError as e:
            logger.error(f"Cannot connect to LDAP server: {e}")
            return False, None
        except LDAPException as e:
            logger.error(f"LDAP authentication error: {e}")
            return False, None

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test the LDAP connection and configuration.

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not LDAP_AVAILABLE:
            return False, "ldap3 library not installed. Run: pip install ldap3"

        if not self.config.enabled:
            return False, "LDAP is not enabled in configuration"

        if not self.config.server_url:
            return False, "LDAP server URL not configured"

        try:
            conn = self._create_connection()
            if conn.bind():
                conn.unbind()
                return True, f"Successfully connected to LDAP server: {self.config.server_url}"
            else:
                return False, f"Failed to bind to LDAP server. Check bind credentials."
        except LDAPSocketOpenError as e:
            return False, f"Cannot connect to LDAP server: {e}"
        except LDAPException as e:
            return False, f"LDAP error: {e}"


# Global LDAP authenticator instance
_ldap_authenticator: Optional[LDAPAuthenticator] = None
_ldap_config_from_db: Optional[Dict[str, Any]] = None


def load_ldap_config_from_db() -> Optional[LDAPConfig]:
    """
    Load LDAP configuration from database.
    Returns None if not found, allowing fallback to environment variables.
    """
    try:
        from src.database.database import SessionLocal
        from src.database.models import SystemSettings

        db = SessionLocal()
        try:
            setting = db.query(SystemSettings).filter(
                SystemSettings.key == "ldap_config"
            ).first()

            if setting and setting.value:
                return LDAPConfig.from_dict(setting.value)
            return None
        finally:
            db.close()
    except Exception as e:
        logger.warning(f"Could not load LDAP config from database: {e}")
        return None


def save_ldap_config_to_db(config: LDAPConfig, user_id: Optional[int] = None) -> bool:
    """
    Save LDAP configuration to database.

    Args:
        config: LDAPConfig to save
        user_id: ID of user making the change (for audit)

    Returns:
        True if saved successfully
    """
    try:
        from src.database.database import SessionLocal
        from src.database.models import SystemSettings
        from datetime import datetime

        db = SessionLocal()
        try:
            setting = db.query(SystemSettings).filter(
                SystemSettings.key == "ldap_config"
            ).first()

            config_data = config.to_dict(include_password=True)

            if setting:
                setting.value = config_data
                setting.updated_at = datetime.utcnow()
                setting.updated_by = user_id
            else:
                setting = SystemSettings(
                    key="ldap_config",
                    value=config_data,
                    description="LDAP authentication configuration",
                    updated_by=user_id
                )
                db.add(setting)

            db.commit()

            # Reset the global authenticator to pick up new config
            reset_ldap_authenticator()

            return True
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Could not save LDAP config to database: {e}")
        return False


def get_ldap_config() -> LDAPConfig:
    """
    Get LDAP configuration.
    Priority: Database config > Environment variables
    """
    # Try database first
    db_config = load_ldap_config_from_db()
    if db_config is not None:
        return db_config

    # Fall back to environment variables
    return LDAPConfig.from_env()


def reset_ldap_authenticator():
    """Reset the global LDAP authenticator to reload configuration."""
    global _ldap_authenticator
    _ldap_authenticator = None


def get_ldap_authenticator() -> LDAPAuthenticator:
    """Get or create the global LDAP authenticator instance."""
    global _ldap_authenticator
    if _ldap_authenticator is None:
        config = get_ldap_config()
        _ldap_authenticator = LDAPAuthenticator(config)
    return _ldap_authenticator


def ldap_authenticate(username: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Convenience function to authenticate a user via LDAP.

    Args:
        username: Username to authenticate
        password: Password to verify

    Returns:
        Tuple of (success: bool, user_info: dict or None)
    """
    return get_ldap_authenticator().authenticate(username, password)


def is_ldap_enabled() -> bool:
    """Check if LDAP authentication is enabled and available."""
    return get_ldap_authenticator().is_available
